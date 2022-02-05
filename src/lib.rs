
use std::fmt::Debug;
use std::io::Cursor;
use derive_more::{Display, Error};
use itertools::Itertools;

use BcError::*;


pub type BcResult<T> = std::result::Result<T, BcError>;


#[derive(Debug, Display, Clone, Copy, Error)]
pub enum BcError {
	#[display(fmt = "ReadError({:?})", _0)]
	ReadError(#[error(ignore)] std::io::ErrorKind),
	ReadEof,

	Waiting,

	#[display(fmt = "WriteError({:?})", _0)]
	WriteError(#[error(ignore)] std::io::ErrorKind),
	WriteEof,

	LzoOutputOverrun,
	LzoLookbehindOverrun,

	RleIncorrectMinimumRunValue,
	RleIncompleteData,
}


/// A general trait that represents a compression or a decompression algorithm.
pub trait Algorithm {
	/// Create a new instance of the algorithm in its initial state (i.e., ready
	/// to read input data from the beginning).
	fn new() -> Self;


	/// Receive one byte of input data, process it, and possibly write one or
	/// more bytes of output data to `output`.  The algorithm may decide that
	/// it may wait for further data; in this case, this method returns
	/// `Err(Waiting)`.  This does not mean that further data is **required**,
	/// only that the algorithm did not have to flush output.
	fn filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize>;

	/// Require the algorithm to flush its internal state, writing the output.
	/// If the algorithm needs to wait for further data because its internal
	/// state is incomplete, it will return an appropriate error.
	#[allow(unused_variables)]
	fn finish<W: Write>(&mut self, output: &mut W) -> BcResult<usize> { Ok(0) }

	fn filter_to_end<R: Read, W: Write>(&mut self, input: &mut R, output: &mut W) -> BcResult<usize> {
		let mut bytes_written: usize = 0;

		loop {
			match input.read_byte() {
				Ok(b) => {
					let result = self.filter_byte::<W>(b, output);

					match result {
						Ok(bytes) => { bytes_written += bytes },
						Err(Waiting) => (),
						Err(e) => return Err(e),
					};
				},

				Err(ReadEof) => {
					bytes_written += self.finish::<W>(output)?;
					return Ok(bytes_written);
				},

				Err(e) => return Err(e),
			};
		}
	}


	fn filter_slice<W: Write>(&mut self, input: &[u8], output: &mut W) -> BcResult<usize> {
		let mut cursor = Cursor::new(input);
		self.filter_to_end(&mut cursor, output)
	}


	fn filter_slice_to_vec(&mut self, input: &[u8]) -> BcResult<Vec<u8>> {
		let mut output = vec![];
		let bytes_written = self.filter_slice(input, &mut output)?;
		output.truncate(bytes_written);
		Ok(output)
	}
}


pub trait Read {
	fn read_byte(&mut self) -> BcResult<u8>;
}


impl<T: std::io::Read> Read for T {
	fn read_byte(&mut self) -> BcResult<u8> {
		let mut buf = [0u8];
		let result = <Self as std::io::Read>::read(self, &mut buf)
			.map_err(|e| {
				match e.kind() {
					std::io::ErrorKind::UnexpectedEof => ReadEof,
					kind => ReadError(kind),
				}
			})?;

		if result == 1 {
			Ok(buf[0])
		} else {
			Err(ReadEof)
		}
	}
}


pub trait Write {
	fn write_byte(&mut self, byte: u8) -> BcResult<()>;
}


impl<T: std::io::Write> Write for T {
	fn write_byte(&mut self, byte: u8) -> BcResult<()> {
		let result = <Self as std::io::Write>::write(self, &[byte])
			.map_err(|e| {
				match e.kind() {
					std::io::ErrorKind::UnexpectedEof => WriteEof,
					kind => WriteError(kind),
				}
			})?;

		if result == 1 {
			Ok(())
		} else {
			Err(WriteEof)
		}
	}
}


#[derive(Debug, Clone, Copy)]
pub enum RleReaderState {
	BlockStart,
	ExpectRleByte(u8),
	ExpectDataBuffer(u8),
}


pub struct RleReader {
	pub state: RleReaderState,
	pub data_buffer: Vec<u8>,
}


impl Algorithm for RleReader {
	fn new() -> Self {
		Self {
			state: RleReaderState::BlockStart,
			data_buffer: Vec::with_capacity(0x80),
		}
	}


	fn filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize> {
		use RleReaderState::*;

		match self.state {
			BlockStart => {
				let is_rle = input & Self::RLE_FLAG != 0;
				let data_size = (input & !Self::RLE_FLAG) + 1;
				self.data_buffer.truncate(0);

				self.state = if is_rle {
					ExpectRleByte(data_size)
				}
				else {
					ExpectDataBuffer(data_size)
				};

				Err(Waiting)
			},

			ExpectRleByte(data_size) => {
				for _ in 0..data_size {
					output.write_byte(input)?;
				};

				self.state = BlockStart;
				Ok(data_size as usize)
			},

			ExpectDataBuffer(data_size) => {
				self.data_buffer.push(input);

				if data_size == self.data_buffer.len() as u8 {
					let bytes_written = self.data_buffer.len() as usize;

					for byte in self.data_buffer.iter() {
						output.write_byte(*byte)?;
					}
					self.data_buffer.truncate(0);

					self.state = BlockStart;
					Ok(bytes_written as usize)
				}
				else {
					Err(Waiting)
				}
			},
		}
	}


	fn finish<W: Write>(&mut self, _: &mut W) -> BcResult<usize> {
		use RleReaderState::*;

		match self.state {
			BlockStart => Ok(0),
			_ => Err(RleIncompleteData),
		}
	}
}


impl RleReader {
	pub const MAX_BLOCK_SIZE: usize = RleWriter::MAX_BLOCK_SIZE;
	pub const RLE_FLAG: u8 = RleWriter::RLE_FLAG;
}


pub struct RleWriter {
	pub data_buffer: Vec<u8>,
	pub current_run: (u8, usize),
	pub minimum_run: usize,
}


impl Algorithm for RleWriter {
	fn new() -> Self {
		Self {
			data_buffer: Vec::with_capacity(Self::MAX_BLOCK_SIZE),
			current_run: (0, 0),
			minimum_run: 2,
		}
	}

	fn filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize> {
		use BcError::*;

		let (ref mut run_byte, ref mut run_size) = self.current_run;

		if input == *run_byte || *run_size == 0 {
			*run_size += 1;
			*run_byte = input;
			return Err(Waiting);
		}

		// input != *run_byte, decide if the RLE run is long enough:
		// - If it is, we have enough information to flush buffers
		// - If not, move RLE run to data_buffer and continue waiting

		let result =
			if *run_size >= self.minimum_run {
				self.finish(output)
			}
			else {
				self.data_buffer.extend(&vec![*run_byte; *run_size][..]);
				Err(Waiting)
			};

		self.current_run = (input, 1);

		result
	}


	fn finish<W: Write>(&mut self, output: &mut W) -> BcResult<usize> {
		let (ref mut run_byte, ref mut run_size) = self.current_run;

		if *run_size < self.minimum_run {
			self.data_buffer.extend(&vec![*run_byte; *run_size][..]);
			*run_size = 0;
		};

		let mut bytes_written: usize = 0;

		for chunk in self.data_buffer.chunks(Self::MAX_BLOCK_SIZE) {
			if chunk.is_empty() {
				continue;
			};

			let header = (chunk.len() - 1) as u8;
			output.write_byte(header)?;

			for b in chunk {
				output.write_byte(*b)?;
			}

			bytes_written += 1 + chunk.len();
		}

		self.data_buffer.truncate(0);

		for chunk in (0..*run_size).chunks(Self::MAX_BLOCK_SIZE).into_iter() {
			let len = chunk.collect_vec().len() as u8;
			let header = (len - 1) ^ Self::RLE_FLAG;
			output.write_byte(header)?;
			output.write_byte(*run_byte)?;
			bytes_written += 2;
		}

		*run_size = 0;

		Ok(bytes_written)
	}
}


impl RleWriter {
	pub const MAX_BLOCK_SIZE: usize = 0x80;
	pub const RLE_FLAG: u8 = 0x80;


	/// Create an RleWriter algorithm instance which requires a contiguous run
	/// of `minimum_run` elements to encode it as RLE.  The default is 2,
	/// TIFF PackBits (which this implementation is similar to, but incompatible
	/// with) uses 3.  `minimum_run` must be in the (inclusive) range of [1, 121].
	pub fn with_minimum_run(minimum_run: usize) -> BcResult<Self> {
		if !(1..=0x79_usize).contains(&minimum_run) {
			return Err(RleIncorrectMinimumRunValue);
		};

		Ok(Self { minimum_run, .. Self::new() })
	}
}


#[test]
fn test_rle() {
	assert_eq!(vec![0u8; 0], RleWriter::new().filter_slice_to_vec(&vec![][..]).unwrap());
	assert_eq!(vec![0x00u8, 0x41], RleWriter::new().filter_slice_to_vec(&vec![0x41u8][..]).unwrap());

	let mut writer = RleWriter::with_minimum_run(2).unwrap();
	let mut reader = RleReader::new();
	let input = vec![0x61, 0x61, 0x84, 0x84, 0x10];
	let wanted = vec![0x81, 0x61, 0x81, 0x84, 0x00, 0x10];
	let mut actual = vec![];
	assert!(matches!(writer.filter_byte(input[0], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[1], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[2], &mut actual), Ok(2)));
	assert!(matches!(writer.filter_byte(input[3], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[4], &mut actual), Ok(2)));
	let result = writer.finish(&mut actual).unwrap();
	assert_eq!(result, 2);
	assert_eq!(wanted, actual);
	let mut uncompressed = vec![];
	let result = reader.filter_to_end(&mut Cursor::new(actual), &mut uncompressed);
	assert!(matches!(result, Ok(5)));
	assert_eq!(input, uncompressed);

	let mut writer = RleWriter::with_minimum_run(3).unwrap();
	let input = vec![0x41, 0x42, 0x42, 0x43, 0x43, 0x43, 0x44, 0x44, 0x45, 0x46, 0x46, 0x46];
	let wanted = vec![0x02, 0x41, 0x42, 0x42, 0x82, 0x43, 0x02, 0x44, 0x44, 0x45, 0x82, 0x46];
	let mut actual = Cursor::new(vec![]);
	assert!(matches!(writer.filter_byte(input[0], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[1], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[2], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[3], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[4], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[5], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[6], &mut actual), Ok(6)));
	assert!(matches!(writer.filter_byte(input[7], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[8], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[9], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[10], &mut actual), Err(Waiting)));
	assert!(matches!(writer.filter_byte(input[11], &mut actual), Err(Waiting)));
	let result = writer.finish(&mut actual).unwrap();
	assert_eq!(6, result);
	assert_eq!(wanted, actual.into_inner());
}
