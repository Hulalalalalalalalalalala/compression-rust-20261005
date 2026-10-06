//! The write-side streaming encoder must survive an output that blocks
//! (`WouldBlock`) halfway through `finish()`: once the output recovers, the
//! caller may `flush()` and `finish()` in any order and must still end up
//! with exactly one complete, uncorrupted frame - no lost bytes, no
//! duplicated bytes, no extra empty frame.

use std::io::{self, Write};

use zstd::stream::raw;
use zstd::stream::write::Encoder;
use zstd::stream::{decode_all, encode_all, zio};

const LEVEL: i32 = 3;

/// Compressible but not trivial input.
fn input() -> Vec<u8> {
    (0..100_000).map(|i| (i % 251) as u8).collect()
}

/// Deterministic incompressible input: the frame trailer zstd has to emit
/// on `finish()` is then much larger than the encoder's output buffer, so a
/// blocked writer interrupts the finish while batches of trailer are still
/// being generated.
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

fn reference(data: &[u8]) -> Vec<u8> {
    encode_all(data, LEVEL).unwrap()
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

/// Keep flushing until it goes through, tolerating `WouldBlock`.
fn flush_through_blocks<W: Write>(w: &mut W) -> io::Result<()> {
    loop {
        match w.flush() {
            Ok(()) => return Ok(()),
            Err(ref e) if would_block(e) => {}
            Err(e) => return Err(e),
        }
    }
}

/// Keep finishing until it goes through, tolerating `WouldBlock`.
fn finish_through_blocks<W: Write>(w: &mut Encoder<'_, W>) -> io::Result<()> {
    loop {
        match w.do_finish() {
            Ok(()) => return Ok(()),
            Err(ref e) if would_block(e) => {}
            Err(e) => return Err(e),
        }
    }
}

/// A finish interrupted by `WouldBlock`, then a `flush()`, must complete the
/// very same frame; further `finish`/`flush` calls must not add any output.
#[test]
fn finish_blocked_then_flush_completes_the_same_frame() {
    let input = noise(300_000);
    let expected = reference(&input);

    let mut encoder =
        Encoder::new(ThrottledWriter::new(usize::MAX), LEVEL).unwrap();
    encoder.write_all(&input).unwrap();

    // Let the finish push a couple of bytes, then block it.
    encoder.get_mut().allow(2);
    let err = encoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    // The output recovers; a flush must drive the interrupted finish to
    // completion rather than corrupting or restarting anything.
    encoder.get_mut().unblock();
    encoder.flush().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    // Finishing again must succeed without emitting anything more.
    encoder.do_finish().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    // Neither must repeated flushes.
    encoder.flush().unwrap();
    encoder.flush().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    // Writing past the finish is rejected.
    assert!(encoder.write(b"nope").is_err());

    // And try_finish hands the writer back, still with no extra output.
    let writer = encoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.data, expected);
    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
}

/// `flush()` before any finish request keeps its usual meaning: push out
/// what has been accepted so far, then keep accepting more input.
#[test]
fn flush_before_finish_still_allows_more_writes() {
    let input = input();
    let (first, second) = input.split_at(input.len() / 2);

    let mut encoder =
        Encoder::new(ThrottledWriter::new(usize::MAX), LEVEL).unwrap();
    encoder.write_all(first).unwrap();
    encoder.flush().unwrap();
    encoder.write_all(second).unwrap();
    let writer = encoder.finish().unwrap();

    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
}

/// With a one-byte output buffer and a one-byte-at-a-time writer, the frame
/// trailer is produced in many batches; blocking and alternating
/// flush/finish in the middle of it must still yield the exact same frame.
#[test]
fn tiny_buffer_finish_blocked_mid_trailer() {
    let input = input();
    let expected = reference(&input);

    let writer = ThrottledWriter::new(1);
    let raw = raw::Encoder::new(LEVEL).unwrap();
    let zio_writer = zio::Writer::new_with_capacity(writer, raw, 1);
    let mut encoder = Encoder::with_writer(zio_writer);

    write_all_through_blocks(&mut encoder, &input).unwrap();

    // Block while the trailer is still being generated in batches.
    encoder.get_mut().allow(3);
    let err = encoder.do_finish().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    // A flush with a small budget blocks again, partway through the trailer.
    encoder.get_mut().allow(2);
    let err = encoder.flush().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    // Once unblocked, flush completes the close...
    encoder.get_mut().unblock();
    encoder.flush().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    // ...and finishing afterwards adds nothing.
    encoder.do_finish().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    let writer = encoder.finish().unwrap();
    assert_eq!(writer.data, expected);
    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
}

/// An output that accepts a single byte at a time and blocks between every
/// two writes: errors come back promptly, and resuming never asks the caller
/// to re-submit already-accepted data.
#[test]
fn one_byte_writer_with_intermittent_wouldblock() {
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
    let expected = reference(&input);

    let mut encoder = Encoder::new(
        Stutter {
            data: Vec::new(),
            accept_next: true,
        },
        LEVEL,
    )
    .unwrap();

    write_all_through_blocks(&mut encoder, &input).unwrap();
    // Alternate flush and finish attempts; each WouldBlock just means
    // "call me again".
    loop {
        match encoder.do_finish() {
            Ok(()) => break,
            Err(ref e) if would_block(e) => match encoder.flush() {
                Ok(()) | Err(_) => {}
            },
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }
    flush_through_blocks(&mut encoder).unwrap();

    let writer = encoder.finish().unwrap();
    assert_eq!(writer.data, expected);
    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
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

    let mut encoder = Encoder::new(ZeroWriter, LEVEL).unwrap();
    encoder.write_all(&input()).unwrap();
    let err = encoder.do_finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::WriteZero);
}

/// Once every compressed byte has been handed over, a failing underlying
/// `flush()` must be reported as-is, and retrying the flush must not make
/// the encoder emit any more frame bytes.
#[test]
fn underlying_flush_error_is_returned_and_retried() {
    let input = input();
    let expected = reference(&input);

    let mut encoder =
        Encoder::new(ThrottledWriter::new(usize::MAX), LEVEL).unwrap();
    encoder.write_all(&input).unwrap();
    encoder.do_finish().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    encoder.get_mut().flush_blocked = true;
    let err = encoder.flush().unwrap_err();
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    encoder.get_mut().flush_blocked = false;
    encoder.flush().unwrap();
    assert_eq!(encoder.get_ref().data, expected);

    let writer = encoder.finish().unwrap();
    assert_eq!(writer.data, expected);
}

/// A failed `try_finish` hands the encoder back along with the error, and
/// the finish can be retried through it.
#[test]
fn try_finish_returns_encoder_on_error() {
    let input = noise(300_000);
    let expected = reference(&input);

    let mut encoder =
        Encoder::new(ThrottledWriter::new(usize::MAX), LEVEL).unwrap();
    encoder.write_all(&input).unwrap();

    encoder.get_mut().allow(0);
    let (mut encoder, err) = match encoder.try_finish() {
        Ok(_) => panic!("try_finish should have failed"),
        Err((encoder, err)) => (encoder, err),
    };
    assert!(would_block(&err), "expected WouldBlock, got {:?}", err);

    encoder.get_mut().unblock();
    let writer = encoder.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.data, expected);
    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
}

/// Empty input must produce exactly one valid empty frame, even through a
/// one-byte output buffer and an interrupted finish.
#[test]
fn empty_input_produces_a_single_empty_frame() {
    let expected = reference(&[]);
    assert!(!expected.is_empty(), "an empty frame is still a frame");

    for capacity in [1, 7, 32 * 1024] {
        let writer = ThrottledWriter::new(1);
        let raw = raw::Encoder::new(LEVEL).unwrap();
        let zio_writer = zio::Writer::new_with_capacity(writer, raw, capacity);
        let mut encoder = Encoder::with_writer(zio_writer);

        // Block the finish right away, then complete it through flush.
        encoder.get_mut().allow(1);
        let err = encoder.do_finish().unwrap_err();
        assert!(would_block(&err), "expected WouldBlock, got {:?}", err);
        encoder.get_mut().unblock();
        encoder.flush().unwrap();
        encoder.do_finish().unwrap();

        let writer = encoder.finish().unwrap();
        assert_eq!(writer.data, expected, "capacity {}", capacity);
        assert_eq!(decode_all(&writer.data[..]).unwrap(), b"");
    }
}

/// Even with a one-byte output buffer, every step makes progress and the
/// final frame is byte-for-byte the reference one.
#[test]
fn one_byte_output_buffer_still_makes_progress() {
    let input = input();
    let expected = reference(&input);

    let writer = ThrottledWriter::new(usize::MAX);
    let raw = raw::Encoder::new(LEVEL).unwrap();
    let zio_writer = zio::Writer::new_with_capacity(writer, raw, 1);
    let mut encoder = Encoder::with_writer(zio_writer);

    encoder.write_all(&input).unwrap();
    finish_through_blocks(&mut encoder).unwrap();

    let writer = encoder.finish().unwrap();
    assert_eq!(writer.data, expected);
    assert_eq!(decode_all(&writer.data[..]).unwrap(), input);
}
