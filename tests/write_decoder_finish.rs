//! The write-side streaming decoder can be explicitly finished to confirm
//! that the compressed input was complete: `finish` hands back the
//! underlying writer on success, `try_finish` hands back the decoder plus
//! the error on failure, and `do_finish` can be retried on the same
//! decoder. A finish only succeeds once the compressed input forms a
//! whole number of frames, every decompressed byte has been delivered to
//! the underlying writer, and the underlying flush succeeded.

use std::io::{self, Write};

use zstd::stream::raw;
use zstd::stream::write::{Decoder, Encoder};
use zstd::stream::{encode_all, zio};

const LEVEL: i32 = 3;

/// Compressible but not trivial input.
fn input() -> Vec<u8> {
    (0..100_000).map(|i| (i % 251) as u8).collect()
}

/// Deterministic incompressible input.
fn noise(len: usize) -> Vec<u8> {
    let mut state = 0x9E3779B97F4A7C15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn frame(data: &[u8]) -> Vec<u8> {
    encode_all(data, LEVEL).unwrap()
}

fn frame_with_checksum(data: &[u8]) -> Vec<u8> {
    let mut encoder = Encoder::new(Vec::new(), LEVEL).unwrap();
    encoder.include_checksum(true).unwrap();
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// A writer that accepts at most `chunk` bytes per `write()` call, and
/// only while `budget` lasts; once the budget is exhausted it returns
/// `WouldBlock` until more budget is allowed.
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

/// A complete frame finishes successfully and hands back the writer with
/// all the decompressed data.
#[test]
fn finish_succeeds_on_complete_frame() {
    let input = input();

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&frame(&input)).unwrap();
    let output = decoder.finish().unwrap();

    assert_eq!(output, input);
}

/// `do_finish` and `try_finish` are the same operation, on `&mut self`
/// and on `self` respectively.
#[test]
fn do_finish_and_try_finish_complete_the_stream() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.do_finish().unwrap();
    let output = decoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(output, input);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(output, input);
}

/// Several complete frames concatenated finish successfully; the output
/// is the concatenation of each frame's content, in order.
#[test]
fn concatenated_frames_finish_in_order() {
    let parts: Vec<Vec<u8>> =
        vec![noise(10_000), input(), b"".to_vec(), b"tiny".to_vec()];
    let expected: Vec<u8> = parts.concat();
    let frames: Vec<Vec<u8>> = parts.iter().map(|p| frame(p)).collect();

    // Frames written one at a time.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    for f in &frames {
        decoder.write_all(f).unwrap();
    }
    assert_eq!(decoder.finish().unwrap(), expected);

    // All frames written in a single call.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&frames.concat()).unwrap();
    assert_eq!(decoder.finish().unwrap(), expected);
}

/// Finishing without any compressed input is an `UnexpectedEof`.
#[test]
fn no_input_is_unexpected_eof() {
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

    // Empty writes are not input either.
    decoder.write_all(b"").unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

    // try_finish reports the same error and hands the decoder back.
    let decoder = Decoder::new(Vec::new()).unwrap();
    let (_decoder, err) = decoder.try_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

/// A valid empty-content frame is complete and finishes successfully.
#[test]
fn empty_frame_finishes_successfully() {
    let compressed = frame(b"");
    assert!(!compressed.is_empty(), "an empty frame is still a frame");

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, b"");
}

/// A truncated last frame - partial header, partial data or partial
/// checksum tail - must be reported as `UnexpectedEof`.
#[test]
fn truncated_frame_is_unexpected_eof() {
    let input = noise(50_000);
    let compressed = frame_with_checksum(&input);

    // Sample prefixes: partial magic, partial header, partial data,
    // and partial checksum tail.
    let cuts = [
        1,
        3,
        5,
        compressed.len() / 2,
        compressed.len() - 3,
        compressed.len() - 1,
    ];
    for cut in cuts {
        let mut decoder = Decoder::new(Vec::new()).unwrap();
        decoder.write_all(&compressed[..cut]).unwrap();
        let err = decoder.do_finish().unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::UnexpectedEof,
            "prefix of {} bytes",
            cut
        );
        // Whatever plaintext was decoded before the cut is still there.
        assert!(input.starts_with(&decoder.get_ref()[..]));
    }
}

/// Complete frames followed by a truncated frame: the completeness of
/// earlier frames does not make up for the truncated last one.
#[test]
fn truncated_last_frame_is_unexpected_eof() {
    let first = frame(b"complete");
    let second = frame(b"truncated");

    for cut in [1, 3, second.len() / 2, second.len() - 1] {
        let mut decoder = Decoder::new(Vec::new()).unwrap();
        decoder.write_all(&first).unwrap();
        decoder.write_all(&second[..cut]).unwrap();
        let err = decoder.do_finish().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof, "cut at {}", cut);
        // The first frame's content was fully delivered, possibly
        // followed by a prefix of the truncated frame's content.
        let output = decoder.get_ref();
        assert!(output.starts_with(b"complete"));
        assert!(b"completetruncated".starts_with(output));
    }
}

/// Writing an empty slice does not start a new frame: a complete last
/// frame stays complete.
#[test]
fn empty_write_does_not_unfinish_the_last_frame() {
    let input = input();

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&frame(&input)).unwrap();
    decoder.write_all(b"").unwrap();
    assert_eq!(decoder.write(b"").unwrap(), 0);
    decoder.flush().unwrap();
    assert_eq!(decoder.finish().unwrap(), input);

    // Also between two frames.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&frame(b"first")).unwrap();
    decoder.write_all(b"").unwrap();
    decoder.write_all(&frame(b"second")).unwrap();
    decoder.write_all(b"").unwrap();
    assert_eq!(decoder.finish().unwrap(), b"firstsecond");
}

/// Corrupted data keeps reporting the decompression error; after such an
/// error, finishing never turns into a success and produces no more
/// plaintext.
#[test]
fn corrupted_data_stays_an_error() {
    // Garbage that is not a zstd frame at all.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    let garbage = [0xDEu8, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8];
    assert!(decoder.write_all(&garbage).is_err());
    assert!(decoder.do_finish().is_err());
    assert!(decoder.get_ref().is_empty());

    // A frame with a corrupted checksum tail.
    let input = input();
    let mut compressed = frame_with_checksum(&input);
    let last = compressed.len() - 1;
    compressed[last] ^= 0xFF;

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    assert!(decoder.write_all(&compressed).is_err());
    // Finishing must not succeed afterwards...
    assert!(decoder.do_finish().is_err());
    // ...and whatever plaintext was delivered stays as-is: a prefix of
    // the original content, never more.
    assert!(input.starts_with(&decoder.get_ref()[..]));
    let delivered = decoder.get_ref().len();
    assert!(decoder.do_finish().is_err());
    assert_eq!(decoder.get_ref().len(), delivered);
}

/// A writer that only accepts a few bytes per call still receives the
/// entire plaintext by the time the finish succeeds.
#[test]
fn short_writes_still_deliver_everything() {
    let input = input();
    let compressed = frame(&input);

    for chunk in [1, 7, 1024] {
        let writer = ThrottledWriter::new(chunk);
        let mut decoder = Decoder::new(writer).unwrap();
        decoder.write_all(&compressed).unwrap();
        let writer = decoder.finish().unwrap();
        assert_eq!(writer.data, input, "chunk {}", chunk);
    }
}

/// A `WouldBlock` from the underlying writer interrupts the finish
/// promptly; once writable again, `do_finish`, `try_finish` or `flush`
/// resume it without losing, duplicating or reordering output.
#[test]
fn finish_blocked_then_resumed() {
    let input = noise(300_000);
    let compressed = frame(&input);

    // Resume through do_finish.
    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.get_mut().allow(2);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    decoder.get_mut().unblock();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref().data, input);

    // Resume through flush, then finish again.
    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.get_mut().allow(3);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    decoder.get_mut().unblock();
    decoder.flush().unwrap();
    decoder.do_finish().unwrap();
    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);

    // Resume through try_finish.
    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.get_mut().allow(1);
    let (mut decoder, err) = match decoder.try_finish() {
        Ok(_) => panic!("try_finish should have failed"),
        Err((decoder, err)) => (decoder, err),
    };
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    decoder.get_mut().unblock();
    let writer = decoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.data, input);
}

/// Once all the plaintext has been delivered, a failing underlying
/// `flush()` is reported as-is, and retrying does not emit any more
/// plaintext.
#[test]
fn underlying_flush_error_is_returned_and_retried() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();

    decoder.get_mut().flush_blocked = true;
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    // The plaintext itself was fully delivered before the flush failed.
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
fn truncated_input_and_blocked_output_reports_output_error_first() {
    let input = noise(100_000);
    let compressed = frame(&input);
    let truncated = &compressed[..compressed.len() / 2];

    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(truncated).unwrap();

    // Block the output: the pending plaintext cannot be delivered, and
    // that error comes first.
    decoder.get_mut().allow(2);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    // Once the output recovers and the pending plaintext is delivered,
    // the truncated input is reported.
    decoder.get_mut().unblock();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

    // The delivered plaintext is a prefix of the original content.
    assert!(input.starts_with(&decoder.get_ref().data[..]));
}

/// A writer that reports `Ok(0)` for non-empty data surfaces as
/// `WriteZero`.
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
    decoder.write_all(&frame(b"hello")).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WriteZero);
}

/// After the first finish request - even one that failed from
/// backpressure - no more compressed input is accepted.
#[test]
fn write_is_rejected_after_finish_is_requested() {
    let input = input();
    let compressed = frame(&input);

    // After a successful finish.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.do_finish().unwrap();
    let err = decoder.write(b"nope").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);
    assert_eq!(decoder.get_ref(), &input);

    // After a finish that failed from backpressure.
    let mut decoder = Decoder::new(ThrottledWriter::new(usize::MAX)).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.get_mut().allow(0);
    let err = decoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
    let err = decoder.write(b"nope").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);

    // The rejected input was not consumed: finishing still yields
    // exactly the original plaintext.
    decoder.get_mut().unblock();
    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);

    // After a finish that failed from truncated input.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder
        .write_all(&compressed[..compressed.len() / 2])
        .unwrap();
    assert!(decoder.do_finish().is_err());
    let err = decoder.write(&compressed[..4]).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);
}

/// Repeating `do_finish` or `flush` after a successful finish succeeds
/// without emitting more plaintext.
#[test]
fn repeated_finish_and_flush_after_success() {
    let input = input();

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&frame(&input)).unwrap();
    decoder.do_finish().unwrap();
    assert_eq!(decoder.get_ref(), &input);

    decoder.do_finish().unwrap();
    decoder.flush().unwrap();
    decoder.flush().unwrap();
    assert_eq!(decoder.get_ref(), &input);

    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}

/// A regular `flush()` before any finish request keeps its usual
/// meaning: push out the current plaintext, then keep accepting input.
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

/// Byte-by-byte input and a one-byte output buffer follow the same
/// rules.
#[test]
fn byte_by_byte_input_with_tiny_output_buffer() {
    let input = input();
    let compressed = frame(&input);

    let writer = ThrottledWriter::new(1);
    let raw = raw::Decoder::new().unwrap();
    let zio_writer = zio::Writer::new_with_capacity(writer, raw, 1);
    let mut decoder = Decoder::with_writer(zio_writer);

    for byte in &compressed {
        decoder.write_all(&[*byte]).unwrap();
    }
    decoder.do_finish().unwrap();

    let writer = decoder.finish().unwrap();
    assert_eq!(writer.data, input);
}

/// A tiny output buffer also reports truncation correctly.
#[test]
fn tiny_output_buffer_truncated_input() {
    let compressed = frame(&input());
    let truncated = &compressed[..compressed.len() - 2];

    let writer = ThrottledWriter::new(usize::MAX);
    let raw = raw::Decoder::new().unwrap();
    let zio_writer = zio::Writer::new_with_capacity(writer, raw, 1);
    let mut decoder = Decoder::with_writer(zio_writer);

    decoder.write_all(truncated).unwrap();
    let err = decoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

/// Dictionaries keep working with the finish API.
#[test]
fn dictionary_still_works() {
    let dictionary = b"some common prefix data used as a dictionary";
    let input = input();

    let mut encoder =
        Encoder::with_dictionary(Vec::new(), LEVEL, dictionary).unwrap();
    encoder.write_all(&input).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut decoder =
        Decoder::with_dictionary(Vec::new(), dictionary).unwrap();
    decoder.write_all(&compressed).unwrap();
    let output = decoder.finish().unwrap();
    assert_eq!(output, input);
}

/// `into_inner` keeps its existing meaning: the writer is returned
/// without any completeness check.
#[test]
fn into_inner_still_works() {
    let input = input();
    let compressed = frame(&input);

    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder.write_all(&compressed).unwrap();
    decoder.flush().unwrap();
    let output = decoder.into_inner();
    assert_eq!(output, input);

    // Even on truncated input, into_inner just hands the writer back.
    let mut decoder = Decoder::new(Vec::new()).unwrap();
    decoder
        .write_all(&compressed[..compressed.len() / 2])
        .unwrap();
    let output = decoder.into_inner();
    assert!(input.starts_with(&output[..]));
}
