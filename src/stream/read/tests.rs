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

#[test]
fn test_flush_reports_bytes_written() {
    let input: Vec<u8> = (0..5000).map(|i| (i % 251) as u8).collect();
    let mut encoder = Encoder::new(&input[..], 1).unwrap();

    // Pull some compressed data out first so there is pending input.
    let mut compressed = Vec::new();
    let mut buf = [0u8; 64];
    let n = encoder.read(&mut buf).unwrap();
    compressed.extend_from_slice(&buf[..n]);

    // Flush with a 1-byte buffer: every return value must fit the buffer
    // and describe exactly the bytes delivered this call.
    let mut one = [0u8; 1];
    loop {
        let n = encoder.flush(&mut one).unwrap();
        assert!(n <= one.len());
        if n == 0 {
            break;
        }
        compressed.extend_from_slice(&one[..n]);
    }

    // The rest of the stream still comes out through `read`.
    encoder.read_to_end(&mut compressed).unwrap();

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_flush_empty_buffer_is_noop() {
    let input = b"some test data, some test data, some test data";
    let mut encoder = Encoder::new(&input[..], 1).unwrap();

    let mut buf = [0u8; 32];
    let n = encoder.read(&mut buf).unwrap();
    let mut compressed = buf[..n].to_vec();

    // No room: reports 0, but must not drop pending output or consume input.
    assert_eq!(encoder.flush(&mut []).unwrap(), 0);
    assert_eq!(encoder.flush(&mut []).unwrap(), 0);

    // A non-empty buffer still delivers everything that was pending.
    let mut one = [0u8; 1];
    loop {
        let n = encoder.flush(&mut one).unwrap();
        if n == 0 {
            break;
        }
        compressed.extend_from_slice(&one[..n]);
    }
    encoder.read_to_end(&mut compressed).unwrap();

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, &input[..]);
}

#[test]
fn test_read_resumes_interrupted_flush() {
    let input: Vec<u8> = (0..20_000).map(|i| (i % 253) as u8).collect();
    let mut encoder = Encoder::new(&input[..], 3).unwrap();

    let mut compressed = Vec::new();
    let mut buf = [0u8; 100];
    let n = encoder.read(&mut buf).unwrap();
    compressed.extend_from_slice(&buf[..n]);

    // Start a flush with a tiny buffer, then switch back to `read` before
    // the flush is done: `read` must deliver the rest of the flushed bytes
    // (ahead of any new input) rather than a premature 0.
    let mut one = [0u8; 1];
    let n = encoder.flush(&mut one).unwrap();
    compressed.extend_from_slice(&one[..n]);

    let mut small = [0u8; 7];
    loop {
        let n = encoder.read(&mut small).unwrap();
        if n == 0 {
            break;
        }
        compressed.extend_from_slice(&small[..n]);
        loop {
            let m = encoder.flush(&mut one).unwrap();
            if m == 0 {
                break;
            }
            compressed.extend_from_slice(&one[..m]);
        }
    }

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, input);
}

/// A source that hands out short chunks, returning `WouldBlock` in between.
struct WouldBlockChunks {
    data: Vec<u8>,
    pos: usize,
    chunk: usize,
    block_next: bool,
}

impl Read for WouldBlockChunks {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        if self.block_next {
            self.block_next = false;
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "not yet"));
        }
        self.block_next = true;
        let n = self.chunk.min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn test_flush_after_would_block() {
    let input: Vec<u8> = (0..3000).map(|i| (i % 241) as u8).collect();
    let source = WouldBlockChunks {
        data: input.clone(),
        pos: 0,
        chunk: 5,
        block_next: false,
    };
    let mut encoder = Encoder::new(source, 1).unwrap();

    let mut compressed = Vec::new();
    let mut buf = [0u8; 16];
    let mut flushed = false;
    loop {
        match encoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => compressed.extend_from_slice(&buf[..n]),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // The read failed, but the input accepted so far must still
                // come out through `flush`.
                let mut three = [0u8; 3];
                loop {
                    let n = encoder.flush(&mut three).unwrap();
                    if n == 0 {
                        break;
                    }
                    flushed = true;
                    compressed.extend_from_slice(&three[..n]);
                }
            }
            Err(e) => panic!("unexpected error: {}", e),
        }
    }
    assert!(flushed);

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_flush_delivers_frame_tail_after_eof() {
    let input: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();
    let mut encoder = Encoder::new(&input[..], 1).unwrap();

    // Read and flush with 1-byte buffers, so the frame tail past EOF ends
    // up delivered through `flush`, one byte at a time.
    let mut compressed = Vec::new();
    let mut one = [0u8; 1];
    loop {
        let mut progressed = false;
        let n = encoder.read(&mut one).unwrap();
        if n > 0 {
            compressed.extend_from_slice(&one[..n]);
            progressed = true;
        }
        loop {
            let m = encoder.flush(&mut one).unwrap();
            if m == 0 {
                break;
            }
            compressed.extend_from_slice(&one[..m]);
            progressed = true;
        }
        if !progressed {
            break;
        }
    }

    // Once the tail is delivered, both entries report a stable 0.
    assert_eq!(encoder.read(&mut one).unwrap(), 0);
    assert_eq!(encoder.flush(&mut one).unwrap(), 0);
    assert_eq!(encoder.read(&mut one).unwrap(), 0);

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_empty_input_produces_empty_frame() {
    let mut encoder = Encoder::new(&[][..], 1).unwrap();
    let mut compressed = Vec::new();
    encoder.read_to_end(&mut compressed).unwrap();
    assert!(!compressed.is_empty());

    let decoded = crate::decode_all(&compressed[..]).unwrap();
    assert!(decoded.is_empty());
}
