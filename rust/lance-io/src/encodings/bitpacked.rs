// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use arrow_array::{Array, ArrayRef, PrimitiveArray, UInt32Array};
use arrow_schema::DataType;
use async_trait::async_trait;
use crate::{
    traits::{Reader, Writer},
    ReadBatchParams,
    Error, Result,
};
use super::{Encoder, Decoder, AsyncIndex};

pub struct BitPackedEncoder<'a> {
    writer: &'a mut dyn Writer,
    bit_width: u8,
}

impl<'a> BitPackedEncoder<'a> {
    pub fn new(writer: &'a mut dyn Writer, bit_width: u8) -> Self {
        Self { writer, bit_width }
    }
}

#[async_trait]
impl<'a> Encoder for BitPackedEncoder<'a> {
    async fn encode(&mut self, array: &[&dyn Array]) -> Result<usize> {
        let pos = self.writer.tell().await?;
        self.writer.write_u8(self.bit_width).await?;

        for arr in array {
            if let Some(primitive_array) = arr.as_any().downcast_ref::<PrimitiveArray<T>>() {
                let values = primitive_array.values();
                let mut buffer = Vec::new();
                let mut current_byte = 0u8;
                let mut bits_written = 0;

                for &value in values {
                    let value = value.to_u64().unwrap();
                    let bits_to_write = std::cmp::min(64 - bits_written, self.bit_width as usize);
                    current_byte |= ((value & ((1 << bits_to_write) - 1)) << bits_written) as u8;
                    bits_written += bits_to_write;

                    if bits_written == 8 {
                        buffer.push(current_byte);
                        current_byte = 0;
                        bits_written = 0;
                    }

                    if bits_to_write < self.bit_width as usize {
                        current_byte = (value >> bits_to_write) as u8;
                        bits_written = self.bit_width as usize - bits_to_write;
                    }
                }

                if bits_written > 0 {
                    buffer.push(current_byte);
                }

                self.writer.write_all(&buffer).await?;
            } else {
                return Err(Error::InvalidInput {
                    source: "BitPackedEncoder only supports primitive arrays".into(),
                    location: location!(),
                });
            }
        }

        Ok(pos)
    }
}

pub struct BitPackedDecoder<'a> {
    reader: &'a dyn Reader,
    position: usize,
    length: usize,
    bit_width: u8,
}

impl<'a> BitPackedDecoder<'a> {
    pub fn new(reader: &'a dyn Reader, position: usize, length: usize, bit_width: u8) -> Self {
        Self {
            reader,
            position,
            length,
            bit_width,
        }
    }
}

#[async_trait]
impl<'a> Decoder for BitPackedDecoder<'a> {
    async fn decode(&self) -> Result<ArrayRef> {
        self.decode_impl(ReadBatchParams::All).await
    }

    async fn take(&self, indices: &UInt32Array) -> Result<ArrayRef> {
        self.decode_impl(ReadBatchParams::Take(indices.clone())).await
    }
}

impl<'a> BitPackedDecoder<'a> {
    async fn decode_impl(&self, params: ReadBatchParams) -> Result<ArrayRef> {
        let (offset, length) = params.offset_length(self.length);
        let total_bits = length * self.bit_width as usize;
        let total_bytes = (total_bits + 7) / 8;

        let mut buffer = vec![0u8; total_bytes];
        self.reader.read_exact_at(self.position + offset * self.bit_width as usize / 8, &mut buffer).await?;

        let mut values = Vec::with_capacity(length);
        let mut current_byte = 0u8;
        let mut bits_read = 0;

        for _ in 0..length {
            let mut value = 0u64;
            let mut bits_to_read = self.bit_width;

            while bits_to_read > 0 {
                if bits_read == 0 {
                    current_byte = buffer[current_byte as usize];
                    bits_read = 8;
                }

                let bits = std::cmp::min(bits_to_read, bits_read);
                let mask = (1 << bits) - 1;
                value |= ((current_byte & mask) as u64) << (self.bit_width - bits_to_read);

                current_byte >>= bits;
                bits_read -= bits;
                bits_to_read -= bits;
            }

            values.push(value);
        }

        Ok(Arc::new(PrimitiveArray::from_vec(values)) as ArrayRef)
    }
}

#[async_trait]
impl<'a> AsyncIndex<ReadBatchParams> for BitPackedDecoder<'a> {
    type Output = Result<ArrayRef>;

    async fn get(&self, params: ReadBatchParams) -> Self::Output {
        self.decode_impl(params).await
    }
}