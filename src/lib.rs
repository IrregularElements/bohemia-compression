#![cfg_attr(feature = "checked_write", feature(min_specialization))]


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

	ArithmeticOverflow,

	#[display(fmt = "WriteError({:?})", _0)]
	WriteError(#[error(ignore)] std::io::ErrorKind),
	WriteEof,

	LzoOutputOverrun,
	LzoLookbehindOverrun,

	RleIncompleteData,
}


/// A general trait that represents a compression or a decompression algorithm.
pub trait Algorithm : Default {
	/// Create a new instance of the algorithm in its initial state (i.e., ready
	/// to read input data from the beginning).
	fn new() -> Self { <Self as Default>::default() }


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

	#[doc(hidden)]
	#[cfg_attr(not(feature = "checked_write"), allow(unused_variables, clippy::let_and_return))]
	fn __filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize> {
		#[cfg(feature = "checked_write")]
		let prev_pos = output.get_pos();

		let result = self.filter_byte(input, output);

		#[cfg(feature = "checked_write")]
		let next_pos = output.get_pos();

		#[cfg(feature = "checked_write")]
		{
			let written = match &result {
				Ok(written) => Some(*written),
				Err(Waiting) => Some(0),
				Err(_) => None,
			};

			if let (Some(prev), Some(next), Some(written)) = (prev_pos, next_pos, written) {
				assert_eq!(prev + written, next,
					"{}::filter_byte returned an incorrect value: returned {}, actually wrote {}",
					std::any::type_name::<Self>(), written, next-prev);
			};
		};

		result
	}


	#[doc(hidden)]
	#[cfg_attr(not(feature = "checked_write"), allow(unused_variables, clippy::let_and_return))]
	fn __finish<W: Write>(&mut self, output: &mut W) -> BcResult<usize> {
		#[cfg(feature = "checked_write")]
		let prev_pos = output.get_pos();

		let result = self.finish(output);

		#[cfg(feature = "checked_write")]
		let next_pos = output.get_pos();

		#[cfg(feature = "checked_write")]
		{
			let written = match &result {
				Ok(written) => Some(*written),
				Err(Waiting) => panic!("{}::finish returned Err(Waiting)", std::any::type_name::<Self>()),
				Err(_) => None,
			};

			if let (Some(prev), Some(next), Some(written)) = (prev_pos, next_pos, written) {
				assert_eq!(prev + written, next,
					"{}::finish returned an incorrect value: returned {}, actually wrote {}",
					std::any::type_name::<Self>(), written, next-prev);
			};
		};

		result
	}


	fn filter_to_end<R: Read, W: Write>(&mut self, input: &mut R, output: &mut W) -> BcResult<usize> {
		let mut bytes_written: usize = 0;

		loop {
			match input.read_byte() {
				Ok(b) => {
					let result = self.__filter_byte::<W>(b, output);

					match result {
						Ok(bytes) => { bytes_written += bytes },
						Err(Waiting) => (),
						Err(e) => return Err(e),
					};
				},

				Err(ReadEof) => {
					bytes_written += self.__finish::<W>(output)?;
					return Ok(bytes_written);
				},

				Err(e) => return Err(e),
			};
		}
	}


	fn filter_iter<I: IntoIterator<Item=u8>, W: Write>(&mut self, input: I, output: &mut W) -> BcResult<usize> {
		let subtotal: usize = <I as IntoIterator>::into_iter(input)
			.map(|byte| self.__filter_byte(byte, output))
			.try_fold(0, |acc: usize, x|
				match x {
					Ok(b) => acc.checked_add(b).ok_or(ArithmeticOverflow),
					Err(Waiting) => Ok(acc),
					e => e,
				})?;

		let total = subtotal.checked_add(self.__finish(output)?).ok_or(ArithmeticOverflow)?;
		Ok(total)
	}


	fn filter_iter_to_vec<I: IntoIterator<Item=u8>>(&mut self, input: I) -> BcResult<Vec<u8>> {
		let mut output = Vec::new();
		let bytes_written = self.filter_iter::<I, Vec<u8>>(input, &mut output)?;
		output.truncate(bytes_written);
		Ok(output)
	}


	fn filter_slice_to_vec(&mut self, input: &[u8]) -> BcResult<Vec<u8>> {
		self.filter_iter_to_vec(input.iter().copied())
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
	#[cfg(feature = "checked_write")]
	fn get_pos(&self) -> Option<usize>;
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


	#[cfg(feature = "checked_write")]
	default fn get_pos(&self) -> Option<usize> { None }
}


#[cfg(feature = "checked_write")]
impl Write for Vec<u8> {
	fn get_pos(&self) -> Option<usize> { Some(self.len()) }
}


#[derive(Debug, Clone, Copy)]
pub enum RleReaderState {
	BlockStart,
	ExpectRleByte(u8),
	ExpectDataBuffer(u8),
}


impl Default for RleReaderState {
	fn default() -> Self { Self::BlockStart }
}


pub struct RleReader {
	pub state: RleReaderState,
	pub data_buffer: Vec<u8>,
}


impl RleReader {
	pub const MAX_BLOCK_SIZE: usize = RleWriter::MAX_BLOCK_SIZE;
	pub const RLE_FLAG: u8 = RleWriter::RLE_FLAG;
}


impl Default for RleReader {
	fn default() -> Self {
		Self {
			state: RleReaderState::BlockStart,
			data_buffer: Vec::with_capacity(0x80),
		}
	}
}


impl Algorithm for RleReader {
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


pub struct RleWriter {
	pub data_buffer: Vec<u8>,
	pub current_run: (u8, usize),
	pub minimum_run: usize,
}


impl RleWriter {
	pub const MAX_BLOCK_SIZE: usize = 0x80;
	pub const RLE_FLAG: u8 = 0x80;


	/// Create an RleWriter algorithm instance which requires a contiguous run
	/// of `minimum_run` elements to encode it as RLE.  The default is 2,
	/// TIFF PackBits (which this implementation is similar to, but incompatible
	/// with) uses 3.  `minimum_run` is clamped to [1, 128].
	pub fn with_minimum_run(minimum_run: usize) -> Self {
		let minimum_run = std::cmp::min(0x80, std::cmp::max(1, minimum_run));

		Self { minimum_run, .. Self::new() }
	}
}


impl Default for RleWriter {
	fn default() -> Self {
		Self {
			data_buffer: Vec::with_capacity(Self::MAX_BLOCK_SIZE),
			current_run: (0, 0),
			minimum_run: 2,
		}
	}
}


impl Algorithm for RleWriter {
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


#[test]
fn test_rle() {
	assert_eq!(vec![0u8; 0], RleWriter::new().filter_slice_to_vec(&vec![][..]).unwrap());
	assert_eq!(vec![0x00u8, 0x41], RleWriter::new().filter_slice_to_vec(&vec![0x41u8][..]).unwrap());

	let mut writer = RleWriter::with_minimum_run(2);
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

	for r in 0..=256 {
		let compressed = RleWriter::with_minimum_run(r).filter_slice_to_vec(&input[..]).unwrap();
		let uncompressed = RleReader::new().filter_slice_to_vec(&compressed[..]).unwrap();
		assert_eq!(input, uncompressed);
	}

	let mut writer = RleWriter::with_minimum_run(3);
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

	for r in 0..=256 {
		let compressed = RleWriter::with_minimum_run(r).filter_slice_to_vec(&input[..]).unwrap();
		let uncompressed = RleReader::new().filter_slice_to_vec(&compressed[..]).unwrap();
		assert_eq!(input, uncompressed);
	}
}


#[derive(Debug)]
enum LzssReaderState {
	Start,
	A,
	B,
}


pub struct LzssReader {
	state: LzssReaderState,
	pub flags: libc::c_uint,
	pub text_buf: [u8; LzssReader::N + LzssReader::F - 1],
	pub i: libc::c_int,
	pub j: libc::c_int,
	pub r: libc::c_int,
	pub c: libc::c_int,
}


impl LzssReader {
	const N: usize = 4096;
	const F: usize = 18;
	const THRESHOLD: usize = 2;
}


impl Default for LzssReader {
	fn default() -> Self {
		let mut result = Self {
			state: LzssReaderState::Start,
			flags: 0,
			text_buf: [0u8; LzssReader::N + LzssReader::F - 1],
			i: 0,
			j: 0,
			r: (Self::N - Self::F) as libc::c_int,
			c: 0,
		};

		for i in result.text_buf.iter_mut().take(Self::N - Self::F) {
			*i = 0x20;
		};

		result
	}
}


impl Algorithm for LzssReader {
	fn filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize> {
		use LzssReaderState::*;

		match &self.state {
			Start => {
				self.flags >>= 1;

				self.state = A;

				if self.flags & 0x100 == 0 {
					self.c = input.into();
					self.flags = libc::c_uint::from_le_bytes(self.c.to_le_bytes()) | 0xFF00;
					Err(Waiting)
				}
				else {
					self.filter_byte(input, output)
				}
			},

			A => {
				if self.flags & 1 != 0 {
					self.c = input.into();
					let value = (self.c & 0xFF) as u8;
					output.write_byte(value)?;
					self.text_buf[self.r as usize] = value;
					self.r += 1;
					self.r &= (Self::N - 1) as libc::c_int;

					self.state = LzssReaderState::Start;
					Ok(1)
				}
				else {
					self.i = input.into();
					self.state = LzssReaderState::B;
					Err(Waiting)
				}
			},

			B => {
				self.j = input.into();
				self.i |= (self.j & 0xF0) << 4;
				self.j &= 0x0F;
				self.j += Self::THRESHOLD as libc::c_int;

				for k in 0..=self.j {
					self.c = self.text_buf[(self.i + k) as usize & (Self::N - 1)] as libc::c_int;
					output.write_byte(self.c as u8)?;
					self.text_buf[self.r as usize] = self.c as u8;
					self.r += 1;
					self.r &= (Self::N - 1) as libc::c_int;
				};

				self.state = LzssReaderState::Start;
				Ok(self.j as usize + 1)
			},
		}
	}
}
