//! The write-side streaming decoder gained an explicit `finish` (with
//! `try_finish`/`do_finish` variants) so callers can confirm that the
//! compressed input ended on a complete frame. Finishing is only successful
//! once the last frame is complete, every decompressed byte has been handed
//! to the underlying writer, and flushing that writer succeeded.

use std::io::{self, Write};

use zstd::stream::raw;
use zstd::stream::write::{Decoder, Encoder};
use zstd::stream::{encode_all, zio};

const LEVEL: i32 = 3;

/// Compressible but not trivial input.
fn input() -> Vec<u8> {
    (0..100_000).map(|i| (i % 251) as u8).collect()
}

/// Input small enough that all its plaintext fits in the decoder's
/// internal output buffer: nothing is forwarded to the underlying writer
/// until it is flushed or finished.
fn small_input() -> Vec<u8> {
    (0..1_000).map(|i| (i % 251) as u8).collect()
}

fn frame(data: &[u8]) -> Vec<u8> {
    encode_all(data, LEVEL).unwrap()
}

/// A frame that carries a content checksum.
fn checksummed_frame(data: &[u8]) -> Vec<u8> {
    let mut encoder = Encoder::new(Vec::new(), LEVEL).unwrap();
    encoder.include_checksum(true).unwrap();
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// A writer that accepts at most `chunk` bytes per `write()` call, and only
/// while `budget` lasts; once the budget is exhausted it returns
/// `WouldBlock` until `unblock()` is called.
struct ThrottledWriter {
    data: Vec<u8>,
    chunk: usize,
    budget: usize,
    flush_blocked: bool,
}

impl ThrottledWriter {
    fn new(chunk: usize) -> Self {
        ThrottledWriter {
            data: Vec::new(),
            chunk,
            budget: usize::MAX,
            flush_blocked: false,
        }
    }

    /// Accept at most `n` more bytes (across all calls), then block.
    fn allow(&mut self, n: usize) {
        self.budget = n;
    }

    fn unblock(&mut self) {
        self.budget = usize::MAX;
    }
}

impl Write for ThrottledWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.budget == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "output is blocked",
            ));
        }
        let n = self.chunk.min(buf.len()).min(self.budget);
        self.data.extend_from_slice(&buf[..n]);
        self.budget -= n;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.flush_blocked {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "flush is blocked",
            ));
        }
        Ok(())
    }
}

fn would_block(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::WouldBlock
}

/// Keep writing until everything has been accepted, tolerating `WouldBlock`.
fn write_all_through_blocks<W: Write>(
    w: &mut W,
    mut buf: &[u8],
) -> io::Result<()> {
    while !buf.is_empty() {
        match w.write(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "zero-length write",
                ))
            }
            Ok(n) => buf = &buf[n..],
            Err(ref e) if would_block(e) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Keep finishing until it goes through, tolerating `WouldBlock`.
fn finish_through_blocks<W: Write>(d: &mut Decoder<'_, W>) -> io::Result<()> {
    loop {
        match d.do_finish() {
            Ok(()) => return Ok(()),
            Err(ref e) if would_block(e) => {}
            Err(e) => return Err(e),
        }
    }
}

/// A complete frame finishes successfully: `do_finish` works, `finish`
/// hands the underlying writer back, and the output is exactly the input.
#[test]
fn finish_succeeds_on_a_complete_frame() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    // Repeating the finish or flushing adds nothing.
    decoder.do_finish().unwrap();
    decoder.flush().unwrap();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// Several complete frames concatenated together finish successfully, and
/// the outputs are concatenated in frame order.
#[test]
fn concatenated_frames_finish_in_order() {
    let parts: Vec<&[u8]> = vec![b"foo", b"bar", b"baz", b""];
    let mut compressed = Vec::new();
    for part in &parts {
        compressed.extend_from_slice(&frame(part));
    }

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, b"foobarbaz");
}

/// No compressed input at all means the stream is incomplete.
#[test]
fn no_input_is_unexpected_eof() {
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

    // An empty write does not change that.
    assert_eq!(decoder.write(&[]).unwrap(), 0);
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

/// A valid frame with empty content is a complete stream.
#[test]
fn empty_frame_finishes_successfully() {
    let compressed = frame(b"");
    assert!(!compressed.is_empty(), "an empty frame is still a frame");

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, b"");
}

/// A last frame that is cut short - in its header, its data, or its
/// checksum trailer - must be reported as `UnexpectedEof`.
#[test]
fn truncated_last_frame_is_unexpected_eof() {
    let input = input();
    let plain = frame(&input);
    let checksummed = checksummed_frame(&input);

    let mut truncations = vec![
        plain[..2].to_vec(),               // partial frame header
        plain[..plain.len() / 2].to_vec(), // partial data
        plain[..plain.len() - 1].to_vec(), // missing last byte
        checksummed[..checksummed.len() - 2].to_vec(), // partial checksum
    ];
    // A complete frame followed by the partial header of the next one.
    let mut almost_two = frame(b"first");
    almost_two.extend_from_slice(&frame(b"second")[..3]);
    truncations.push(almost_two);

    for compressed in truncations {
        let mut decoder = Decoder::new(Vec::new()).unwrap();
        decoder.write_all(&compressed).unwrap();
        let err = decoder.do_finish().unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::UnexpectedEof,
            "truncated input {:?}",
            &compressed[..compressed.len().min(8)]
        );
        // It keeps failing on retry.
        let err = decoder.do_finish().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}

/// The plaintext of the complete frames before a truncated last frame must
/// have been delivered; only the incomplete tail is missing.
#[test]
fn complete_frames_before_a_truncated_one_are_kept() {
    let mut compressed = frame(b"first");
    compressed.extend_from_slice(&frame(b"second")[..5]);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(decoder.get_ref(), b"first");
}

/// Writing an empty slice does not start a new frame: a complete last frame
/// stays complete.
#[test]
fn empty_write_does_not_unfinish_a_frame() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    assert_eq!(decoder.write(&[]).unwrap(), 0);
    decoder.write_all(&[]).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}

/// Corrupt data keeps reporting the decompression error; once it has been
/// seen, finishing never succeeds and no more plaintext is produced.
#[test]
fn corrupt_data_stays_an_error() {
    let first = frame(b"first");
    let mut second = checksummed_frame(b"second");
    // Corrupt the checksum of the second frame.
    let last = second.len() - 1;
    second[last] ^= 0xFF;

    let mut compressed = first.clone();
    compressed.extend_from_slice(&second);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    // The corruption is reported through the existing decompression error.
    assert!(decoder.write_all(&compressed).is_err());
    // The first frame's plaintext was delivered before the error.
    assert_eq!(decoder.get_ref(), b"first");

    // Finishing must not succeed, and must not produce more plaintext.
    assert!(decoder.do_finish().is_err());
    assert_eq!(decoder.get_ref(), b"first");
    assert!(decoder.do_finish().is_err());
    assert!(decoder.flush().is_err());
    assert_eq!(decoder.get_ref(), b"first");

    // try_finish hands the decoder back, still with the error.
    let (decoder, _) = match decoder.try_finish() {
        Ok(_) => panic!("try_finish should have failed"),
        Err((decoder, err)) => (decoder, err),
    };
    assert_eq!(decoder.get_ref(), b"first");
}

/// A bad checksum is a decompression error, not an `UnexpectedEof`.
#[test]
fn bad_checksum_is_a_decompression_error() {
    let mut compressed = checksummed_frame(b"some checksummed content");
    let last = compressed.len() - 1;
    compressed[last] ^= 0xFF;

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    let result = decoder.write_all(&compressed);
    let write_failed = result.is_err();
    let err = match decoder.do_finish() {
        Ok(()) => panic!("a bad checksum must not finish successfully"),
        Err(err) => err,
    };
    assert!(
        write_failed || err.kind() != io::ErrorKind::UnexpectedEof,
        "a checksum mismatch is a corruption error, not a truncation"
    );
    // And it stays an error.
    assert!(decoder.do_finish().is_err());
}

/// A finish interrupted by `WouldBlock` delivers the rest of the plaintext
/// on retry - nothing lost, nothing duplicated, nothing reordered.
#[test]
fn finish_blocked_then_resume_delivers_everything_once() {
    let input = small_input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();

    // Let the finish push a couple of bytes, then block it.
    decoder.get_mut().allow(2);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    assert_eq!(decoder.get_ref().data.len(), 2);

    // Once unblocked, the retry delivers exactly the remaining plaintext.
    decoder.get_mut().unblock();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    // Repeating adds nothing.
    decoder.do_finish().unwrap();
    decoder.flush().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// A `flush()` after an interrupted finish drives the finish to completion.
#[test]
fn flush_completes_an_interrupted_finish() {
    let input = small_input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();

    decoder.get_mut().allow(1);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    decoder.get_mut().unblock();
    decoder.flush().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    decoder.do_finish().unwrap();
    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// Once every plaintext byte has been handed over, a failing underlying
/// `flush()` is reported as-is, and retrying does not emit any more output.
#[test]
fn underlying_flush_error_is_returned_and_retried() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();

    decoder.get_mut().flush_blocked = true;
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    // All the plaintext was delivered before the flush failed.
    assert_eq!(decoder.get_ref().data, input);

    decoder.get_mut().flush_blocked = false;
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// With a truncated input and a temporarily blocked output, the output
/// error is reported first; once the pending plaintext is delivered, the
/// truncation surfaces as `UnexpectedEof`.
#[test]
fn truncated_input_and_blocked_output_reports_output_first() {
    let input = small_input();
    let compressed = checksummed_frame(&input);
    // Cut the checksum trailer: every plaintext byte still decodes, only
    // the frame end is missing.
    let truncated = &compressed[..compressed.len() - 2];

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(truncated).unwrap();

    // The output is blocked: the finish reports the output error first.
    decoder.get_mut().allow(0);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    assert!(decoder.get_ref().data.is_empty());

    // Once the output recovers and the pending plaintext is delivered...
    decoder.get_mut().unblock();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    // ...the decompressed prefix has been fully delivered.
    assert_eq!(decoder.get_ref().data, input);
}

/// A writer that reports `Ok(0)` for non-empty data must surface as
/// `WriteZero`, not as a busy loop or a silent success.
#[test]
fn zero_length_write_is_reported() {
    struct ZeroWriter;

    impl Write for ZeroWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Ok(0)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut decoder = Decoder::new(ZeroWriter).unwrap();
    // Small enough that the plaintext stays in the internal buffer:
    // the refusal only surfaces when the finish tries to deliver it.
    decoder.write_all(&frame(&small_input())).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WriteZero);
}

/// Once a finish has been requested - even one that failed on backpressure -
/// new non-empty compressed input is rejected and not consumed.
#[test]
fn write_after_finish_request_is_rejected() {
    let input = input();
    let compressed = frame(&input);

    // After a finish interrupted by backpressure.
    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.get_mut().allow(0);
    assert!(would_block(&decoder.do_finish().unwrap_err()));

    let err = decoder.write(b"nope").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);
    // Empty writes carry no input and stay a no-op.
    assert_eq!(decoder.write(&[]).unwrap(), 0);

    // The finish can still complete afterwards.
    decoder.get_mut().unblock();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    // After a successful finish too.
    let err = decoder.write(b"nope").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);
    assert_eq!(decoder.get_ref().data, input);
}

/// A regular `flush()` before any finish request just pushes out the
/// current plaintext; more compressed input can still be written after it.
#[test]
fn flush_before_finish_still_allows_more_writes() {
    let input = input();
    let compressed = frame(&input);
    let (first, second) = compressed.split_at(compressed.len() / 2);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(first).unwrap();
    decoder.flush().unwrap();
    decoder.write_all(second).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}

/// A failed `try_finish` hands the decoder back along with the error, and
/// the finish can be retried through it.
#[test]
fn try_finish_returns_decoder_on_error() {
    let input = small_input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();

    decoder.get_mut().allow(0);
    let (mut decoder, err) = match decoder.try_finish() {
        Ok(_) => panic!("try_finish should have failed"),
        Err((decoder, err)) => (decoder, err),
    };
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    decoder.get_mut().unblock();
    let writer = decoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.data, input);
}

/// Feeding the compressed stream one byte at a time through a one-byte
/// output buffer and a one-byte-at-a-time writer yields the same results.
#[test]
fn byte_by_byte_input_with_tiny_output_buffer() {
    let input = input();
    let compressed = frame(&input);

    let writer = ThrottledWriter::new(1);
    let raw = raw::Decoder::new().unwrap();
    let zio_writer = zio::Writer::new_with_capacity(writer, raw, 1);
    let mut decoder = Decoder::with_writer(zio_writer);

    for byte in compressed.chunks(1) {
        write_all_through_blocks(&mut decoder, byte).unwrap();
    }

    finish_through_blocks(&mut decoder).unwrap();
    assert_eq!(decoder.get_ref().data, input);

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// A one-byte-at-a-time writer that blocks between every two writes: the
/// finish just has to be retried until it goes through.
#[test]
fn stuttering_writer_eventually_finishes() {
    struct Stutter {
        data: Vec<u8>,
        accept_next: bool,
    }

    impl Write for Stutter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.accept_next && !buf.is_empty() {
                self.data.push(buf[0]);
                self.accept_next = false;
                Ok(1)
            } else {
                self.accept_next = true;
                Err(io::Error::new(io::ErrorKind::WouldBlock, "stutter"))
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(Stutter {
        data: Vec::new(),
        accept_next: true,
    })
    .unwrap();

    write_all_through_blocks(&mut decoder, &compressed).unwrap();
    loop {
        match decoder.do_finish() {
            Ok(()) => break,
            Err(ref e) if would_block(e) => match decoder.flush() {
                Ok(()) | Err(_) => {}
            },
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// Dictionary-based decoding keeps working with the explicit finish.
#[test]
fn finish_with_dictionary() {
    let dict = b"the quick brown fox jumps over the lazy dog";
    let input = b"the quick brown fox jumps over the lazy dog!";

    let mut encoder =
        Encoder::with_dictionary(Vec::new(), LEVEL, dict).unwrap();
    encoder.write_all(input).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder = Decoder::with_dictionary(Vec::new(), dict).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}

/// The pre-existing entry points are untouched: `into_inner` and
/// `auto_flush` keep their meaning.
#[test]
fn into_inner_and_auto_flush_still_work() {
    let input = input();
    let compressed = frame(&input);

    // into_inner: no completeness check, just the writer back.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.flush().unwrap();
    let output = decoder.into_inner();
    assert_eq!(output, input);

    // auto_flush: dropping flushes the pending plaintext.
    let mut output = Vec::new();
    {
        let mut decoder = Decoder::new(&mut output).unwrap().auto_flush();
        decoder.write_all(&compressed).unwrap();
    }
    assert_eq!(output, input);

    // A decoder that is finished explicitly can also be dropped normally.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}
