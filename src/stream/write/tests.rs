use std::io::{Cursor, Write};
use std::iter;

use partial_io::{PartialOp, PartialWrite};

use crate::stream::decode_all;
use crate::stream::write::{Decoder, Encoder};

/// A writer that only accepts a limited number of bytes per call, and only
/// while it has remaining budget; it then fails with `WouldBlock` until the
/// test tops the budget up again.
struct BlockableWriter {
    buf: Vec<u8>,
    /// Bytes accepted per single `write` call.
    limit: usize,
    /// Total bytes that may still be accepted before `WouldBlock`.
    budget: usize,
    /// When `true`, `flush` fails with `WouldBlock`.
    flush_blocked: bool,
}

impl BlockableWriter {
    fn new(limit: usize) -> Self {
        BlockableWriter {
            buf: Vec::new(),
            limit,
            budget: usize::MAX,
            flush_blocked: false,
        }
    }
}

impl Write for BlockableWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if self.budget == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "blocked",
            ));
        }
        let n = data.len().min(self.limit).min(self.budget);
        self.buf.extend_from_slice(&data[..n]);
        self.budget -= n;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.flush_blocked {
            Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "blocked",
            ))
        } else {
            Ok(())
        }
    }
}

/// Pseudo-random, hard-to-compress data, so the frame tail produced by
/// `finish` spans many internal buffer fills.
fn noisy_input(len: usize) -> Vec<u8> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// Encode `input` without any interruption, as a reference output.
fn reference_encode(input: &[u8]) -> Vec<u8> {
    let mut encoder = Encoder::new(Vec::new(), 1).unwrap();
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

#[test]
fn test_cycle() {
    let input = b"Abcdefghabcdefgh";

    let buffer = Cursor::new(Vec::new());
    let mut encoder = Encoder::new(buffer, 1).unwrap();
    encoder.write_all(input).unwrap();
    let encoded = encoder.finish().unwrap().into_inner();

    // println!("Encoded: {:?}", encoded);

    let buffer = Cursor::new(Vec::new());
    let mut decoder = Decoder::new(buffer).unwrap();
    decoder.write_all(&encoded).unwrap();
    decoder.flush().unwrap();
    let decoded = decoder.into_inner().into_inner();

    assert_eq!(input, &decoded[..]);
}

/// Test that flush after a partial write works successfully without
/// corrupting the frame. This test is in this module because it checks
/// internal implementation details.
#[test]
fn test_partial_write_flush() {
    let input = vec![b'b'; 128 * 1024];
    let mut z = setup_partial_write(&input);

    // flush shouldn't corrupt the stream
    z.flush().unwrap();

    let buf = z.finish().unwrap().into_inner();
    assert_eq!(&decode_all(&buf[..]).unwrap(), &input);
}

/// Test that finish after a partial write works successfully without
/// corrupting the frame. This test is in this module because it checks
/// internal implementation details.
#[test]
fn test_partial_write_finish() {
    let input = vec![b'b'; 128 * 1024];
    let z = setup_partial_write(&input);

    // finish shouldn't corrupt the stream
    let buf = z.finish().unwrap().into_inner();
    assert_eq!(&decode_all(&buf[..]).unwrap(), &input);
}

fn setup_partial_write(
    input_data: &[u8],
) -> Encoder<'_, PartialWrite<Vec<u8>>> {
    let buf =
        PartialWrite::new(Vec::new(), iter::repeat(PartialOp::Limited(1)));
    let mut z = Encoder::new(buf, 1).unwrap();

    // Fill in enough data to make sure the buffer gets written out.
    z.write_all(input_data).unwrap();

    {
        let inner = &mut z.writer;
        // At this point, the internal buffer in z should have some data.
        assert_ne!(inner.offset(), inner.buffer().len());
    }

    z
}

/// Interrupt `do_finish` with `WouldBlock`, then alternate `flush` and
/// `do_finish` while the output only accepts a few bytes at a time.
/// The result must be the exact same frame as an uninterrupted encode.
#[test]
fn test_blocked_finish_then_alternating_flush_and_finish() {
    let input = noisy_input(200 * 1024);
    let expected = reference_encode(&input);

    let mut z = Encoder::new(BlockableWriter::new(7), 1).unwrap();
    z.write_all(&input).unwrap();

    // Block the output entirely: finishing must fail promptly...
    z.get_mut().budget = 0;
    assert_eq!(
        z.do_finish().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    // ...and the encoder must not accept more data.
    assert!(z.write(b"more data").is_err());

    // Resume a few bytes at a time, alternating flush and do_finish.
    let mut use_flush = true;
    loop {
        z.get_mut().budget += 3;
        let result = if use_flush { z.flush() } else { z.do_finish() };
        use_flush = !use_flush;
        match result {
            Ok(()) => break,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
            Err(e) => panic!("unexpected error: {}", e),
        }
    }

    // The whole frame must have been emitted, byte for byte.
    assert_eq!(z.get_ref().buf, expected);

    // Finishing again must succeed without adding any output.
    z.do_finish().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    // So must repeated flushes.
    z.flush().unwrap();
    z.flush().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    // And `try_finish` must return the writer, still without new output.
    let writer = z.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.buf, expected);

    assert_eq!(&decode_all(&expected[..]).unwrap(), &input);
}

/// A `flush` after the close was requested must complete that close
/// (not restart compression), and later finishes must be no-ops.
#[test]
fn test_flush_completes_requested_finish() {
    let input = noisy_input(100 * 1024);
    let expected = reference_encode(&input);

    let mut z = Encoder::new(BlockableWriter::new(5), 1).unwrap();
    z.write_all(&input).unwrap();

    // Interrupt the finish almost immediately.
    z.get_mut().budget = 1;
    assert_eq!(
        z.do_finish().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );

    // A flush must drive the close to completion.
    z.get_mut().budget = usize::MAX;
    z.flush().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    // Everything after that is a no-op producing no output.
    z.do_finish().unwrap();
    z.flush().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    let writer = z.try_finish().map_err(|(_, e)| e).unwrap();
    assert_eq!(writer.buf, expected);
}

/// If the underlying `flush` fails after all compressed data was sent,
/// the error must be returned, and retrying must not emit more frame data.
#[test]
fn test_flush_error_after_close_does_not_regenerate_frame() {
    let input = noisy_input(50 * 1024);
    let expected = reference_encode(&input);

    let mut z = Encoder::new(BlockableWriter::new(1024), 1).unwrap();
    z.write_all(&input).unwrap();
    z.do_finish().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    // Underlying flush fails: the error must bubble up...
    z.get_mut().flush_blocked = true;
    assert_eq!(
        z.flush().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    // ...and retrying must only retry the flush, not regenerate the frame.
    z.get_mut().flush_blocked = false;
    z.flush().unwrap();
    assert_eq!(z.get_ref().buf, expected);
}

/// An encoder with a 1-byte output buffer must still make progress.
#[test]
fn test_tiny_output_buffer() {
    let input = noisy_input(10 * 1024);

    let operation = crate::stream::raw::Encoder::new(1).unwrap();
    let writer = crate::stream::zio::Writer::new_with_capacity(
        Vec::new(),
        operation,
        1,
    );
    let mut z = Encoder::with_writer(writer);
    z.write_all(&input).unwrap();
    let encoded = z.finish().unwrap();

    assert_eq!(&decode_all(&encoded[..]).unwrap(), &input);
}

/// An empty input must produce exactly one valid empty frame.
#[test]
fn test_empty_input_single_empty_frame() {
    let expected = reference_encode(b"");

    let mut z = Encoder::new(BlockableWriter::new(1), 1).unwrap();
    // Interrupt the (empty) finish a few times.
    z.get_mut().budget = 0;
    assert_eq!(
        z.do_finish().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    z.get_mut().budget += 1;
    while z.do_finish().is_err() {
        z.get_mut().budget += 1;
    }
    assert_eq!(z.get_ref().buf, expected);

    // No extra empty frame may be appended by further flushes/finishes.
    z.flush().unwrap();
    z.do_finish().unwrap();
    assert_eq!(z.get_ref().buf, expected);

    assert!(decode_all(&expected[..]).unwrap().is_empty());
}
