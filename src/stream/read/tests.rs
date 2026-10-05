use crate::stream::read::{Decoder, Encoder};
use std::io::{self, Read};

#[test]
fn test_error_handling() {
    let invalid_input = b"Abcdefghabcdefgh";

    let mut decoder = Decoder::new(&invalid_input[..]).unwrap();
    let output = decoder.read_to_end(&mut Vec::new());

    assert!(output.is_err());
}

#[test]
fn test_cycle() {
    let input = b"Abcdefghabcdefgh";

    let mut encoder = Encoder::new(&input[..], 1).unwrap();
    let mut buffer = Vec::new();
    encoder.read_to_end(&mut buffer).unwrap();

    let mut decoder = Decoder::new(&buffer[..]).unwrap();
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer).unwrap();

    assert_eq!(input, &buffer[..]);
}

#[test]
fn test_read_to_end_over_several_blocks() {
    // The `NoOp` test in `zio::reader` covers the buffer bookkeeping on its
    // own. This puts real block boundaries under it, which is what the
    // chunked initialization actually has to interleave with.
    let block = zstd_safe::BLOCKSIZE_MAX as usize;
    let input: Vec<u8> =
        (0..block * 2 + 1234).map(|i| (i % 251) as u8).collect();
    let compressed = crate::bulk::compress(&input, 1).unwrap();

    // Vary what the destination already holds, and how much room it has: the
    // implementation treats "no spare capacity" and "some spare capacity"
    // differently, and has to append rather than overwrite either way.
    for spare in [0, 1, block, input.len() + 1] {
        let mut output = Vec::with_capacity(6 + spare);
        output.extend_from_slice(b"prefix");

        let mut decoder = Decoder::new(&compressed[..]).unwrap();
        let read = decoder.read_to_end(&mut output).unwrap();

        assert_eq!(read, input.len(), "spare {}", spare);
        assert_eq!(&output[..6], b"prefix", "spare {}", spare);
        assert_eq!(&output[6..], &input[..], "spare {}", spare);

        // Reading again after the end appends nothing.
        assert_eq!(decoder.read_to_end(&mut output).unwrap(), 0);
        assert_eq!(output.len(), 6 + input.len());
    }
}

#[test]
fn test_read_to_end_leaves_no_padding_on_error() {
    // A frame that stops early: whatever was decoded stays, but none of the
    // space the implementation initialized may show up as data.
    let input = vec![7u8; 100_000];
    let compressed = crate::bulk::compress(&input, 1).unwrap();
    let truncated = &compressed[..compressed.len() / 2];

    // Spare capacity matters: with none, the implementation probes with a
    // stack buffer and never initializes anything. It is the resize path that
    // could leave zeros behind.
    let mut output = Vec::with_capacity(6 + input.len());
    output.extend_from_slice(b"prefix");
    let mut decoder = Decoder::new(truncated).unwrap();

    assert!(decoder.read_to_end(&mut output).is_err());
    assert_eq!(&output[..6], b"prefix");
    assert!(
        input.starts_with(&output[6..]),
        "padding leaked into the output"
    );
}

/// A reader that hands out its data in short chunks, and once its current
/// burst is spent fails with `WouldBlock` until given a new burst.
struct Stutter<'a> {
    data: &'a [u8],
    chunk: usize,
    burst: usize,
    sealed: bool,
}

impl<'a> Stutter<'a> {
    fn new(data: &'a [u8], chunk: usize) -> Self {
        Stutter {
            data,
            chunk,
            burst: 0,
            sealed: false,
        }
    }
}

impl Read for Stutter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        assert!(!self.sealed, "reader touched after being sealed");
        if self.data.is_empty() {
            // Real EOF.
            return Ok(0);
        }
        if self.burst == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "not ready",
            ));
        }
        let n = self
            .chunk
            .min(self.burst)
            .min(self.data.len())
            .min(buf.len());
        let (head, tail) = self.data.split_at(n);
        buf[..n].copy_from_slice(head);
        self.data = tail;
        self.burst -= n;
        Ok(n)
    }
}

#[test]
fn test_flush_interleaved_with_read_roundtrip() {
    let input: Vec<u8> = (0..50_000).map(|i| (i % 251) as u8).collect();

    let mut encoder = Encoder::new(Stutter::new(&input, 137), 3).unwrap();
    let mut compressed = Vec::new();
    let mut byte = [0u8; 1];
    let mut buf = [0u8; 13];

    // Read until the source runs out of burst and blocks.
    macro_rules! read_until_blocked {
        () => {
            loop {
                match encoder.read(&mut buf) {
                    Ok(0) => panic!("read must not report EOF yet"),
                    Ok(n) => compressed.extend_from_slice(&buf[..n]),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(e) => panic!("unexpected error: {}", e),
                }
            }
        };
    }

    // Drain pending output through flush, one byte at a time.
    macro_rules! flush_all {
        () => {
            loop {
                let n = encoder.flush(&mut byte).unwrap();
                assert!(n <= byte.len());
                if n == 0 {
                    break;
                }
                compressed.extend_from_slice(&byte[..n]);
            }
        };
    }

    // Feed a first burst: read pulls until the source blocks. The error
    // must not lose the input already accepted.
    encoder.get_mut().get_mut().burst = 10_000;
    read_until_blocked!();

    // The consumed input flushes out without touching the source, one
    // byte at a time. An empty buffer is a no-op at any point.
    assert_eq!(encoder.flush(&mut []).unwrap(), 0);
    flush_all!();

    // Feed a second burst, then interrupt the flush after a single byte:
    // read must deliver the rest of that pending output before taking
    // more input, and must not report EOF.
    encoder.get_mut().get_mut().burst = 20_000;
    read_until_blocked!();
    let n = encoder.flush(&mut byte).unwrap();
    assert_eq!(n, 1);
    compressed.extend_from_slice(&byte[..n]);
    read_until_blocked!();
    flush_all!();

    // Feed the rest and let read reach the real EOF. The frame tail may
    // then come out through either entry point, in call order.
    encoder.get_mut().get_mut().burst = usize::MAX;
    loop {
        let n = encoder.read(&mut byte).unwrap();
        compressed.extend_from_slice(&byte[..n]);
        let m = encoder.flush(&mut byte).unwrap();
        compressed.extend_from_slice(&byte[..m]);
        if n == 0 && m == 0 {
            break;
        }
    }

    // Both entry points are done and must not touch the source anymore.
    encoder.get_mut().get_mut().sealed = true;
    assert_eq!(encoder.read(&mut buf).unwrap(), 0);
    assert_eq!(encoder.flush(&mut buf).unwrap(), 0);
    assert_eq!(encoder.flush(&mut []).unwrap(), 0);

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_flush_return_value_matches_delivery() {
    let input: Vec<u8> = (0..20_000).map(|i| (i % 253) as u8).collect();

    // An encoder with the same 5,000 bytes of input already consumed,
    // blocked before it can reach EOF.
    let make = || {
        let mut source = Stutter::new(&input, 91);
        source.burst = 5_000;
        let mut encoder = Encoder::new(source, 3).unwrap();
        let mut buf = [0u8; 64];
        loop {
            match encoder.read(&mut buf) {
                Ok(0) => panic!("read must not report EOF yet"),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
        encoder
    };

    // Drain one a byte at a time, the other in larger slices: both must
    // deliver the same bytes, and every return value must be the number
    // of bytes actually written - including the last, partial batch.
    let mut enc_small = make();
    let mut small = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = enc_small.flush(&mut byte).unwrap();
        assert!(n <= byte.len(), "flush returned more than it wrote");
        if n == 0 {
            break;
        }
        small.extend_from_slice(&byte[..n]);
    }

    let mut enc_big = make();
    let mut big = Vec::new();
    let mut buf = [0u8; 64];
    loop {
        let n = enc_big.flush(&mut buf).unwrap();
        assert!(n <= buf.len(), "flush returned more than it wrote");
        if n == 0 {
            break;
        }
        big.extend_from_slice(&buf[..n]);
    }

    assert!(!small.is_empty());
    assert_eq!(small, big);
}

#[test]
fn test_flush_empty_input_gives_empty_frame() {
    // Empty input still yields a complete (empty) frame, and an empty
    // output buffer never changes anything.
    let input: &[u8] = b"";
    let mut encoder = Encoder::new(input, 1).unwrap();

    assert_eq!(encoder.flush(&mut []).unwrap(), 0);

    let mut compressed = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = encoder.read(&mut byte).unwrap();
        compressed.extend_from_slice(&byte[..n]);
        let m = encoder.flush(&mut byte).unwrap();
        compressed.extend_from_slice(&byte[..m]);
        if n == 0 && m == 0 {
            break;
        }
    }
    assert_eq!(encoder.flush(&mut []).unwrap(), 0);

    assert!(!compressed.is_empty(), "empty input still produces a frame");
    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(&decoded[..], input);
}

#[test]
fn test_flush_after_complete_frame_returns_zero() {
    let input = b"Abcdefghabcdefgh";
    let mut encoder = Encoder::new(&input[..], 1).unwrap();
    let mut compressed = Vec::new();
    encoder.read_to_end(&mut compressed).unwrap();

    let mut buf = [0u8; 8];
    assert_eq!(encoder.flush(&mut buf).unwrap(), 0);
    assert_eq!(encoder.read(&mut buf).unwrap(), 0);

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(&decoded[..], &input[..]);
}
