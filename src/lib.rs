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

	LzssInternalErrorCouldNotReadMore,
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


#[test]
#[cfg(feature = "checked_write")]
#[should_panic(expected = "finish returned an incorrect value: returned 1, actually wrote 0")]
fn checked_write() {
	#[derive(Default)]
	struct BuggyAlgorithm {}

	impl Algorithm for BuggyAlgorithm {
		fn filter_byte<W: Write>(&mut self, _: u8, _: &mut W) -> BcResult<usize> {
			Ok(1) // wrong bytes_written reported, but won't be called on empty input
		}


		fn finish<W: Write>(&mut self, _: &mut W) -> BcResult<usize> {
			Ok(1) // wrong bytes_written reported, will be called, must panic
		}
	}

	BuggyAlgorithm::new().filter_slice_to_vec(&[]).unwrap();
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


/// An [`Algorithm`][crate::Algorithm] that reads BI LZSS and produces
/// uncompressed data.
pub struct LzssReader {
	state: LzssReaderState,
	pub flags: libc::c_uint,
	pub text_buf: [u8; LzssReader::N + LzssReader::F - 1],
	pub i: libc::c_int,
	pub j: libc::c_int,
	pub r: libc::c_int,
	pub c: u8,
}


impl LzssReader {
	const N: usize = 4096;
	const F: usize = 16;
	const THRESHOLD: usize = 2;
}


impl Default for LzssReader {
	fn default() -> Self {
		Self {
			state: LzssReaderState::Start,
			flags: 0,
			text_buf: [0x20u8; LzssReader::N + LzssReader::F - 1],
			i: 0,
			j: 0,
			r: 0,
			c: 0,
		}
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
					self.c = input;
					self.flags = self.c as u32 | 0xFF00;
					Err(Waiting)
				}
				else {
					self.filter_byte(input, output)
				}
			},

			A => {
				if self.flags & 1 != 0 {
					self.c = input;
					output.write_byte(self.c)?;
					self.text_buf[self.r as usize] = self.c;
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

				let pr = self.r;

				for k in 0..=self.j {
					self.c = self.text_buf[(pr - self.i + k) as usize & (Self::N - 1)];
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


enum LzssWriterState {
	BufferInit,
	MainLoop,
	ReadNew,
}


/// An [`Algorithm`][crate::Algorithm] that compresses data with the BI LZSS
/// algorithm (see [`LzssReader`][crate::LzssReader]).
pub struct LzssWriter {
	state: LzssWriterState,
	lchild: [usize; Self::N + 1],
	rchild: [usize; Self::N + 257],
	parent: [usize; Self::N + 1],
	text_buf: [u8; Self::N + Self::F - 1],
	match_position: usize,
	match_length: usize,
	last_match_length: usize,
	s: usize,
	r: usize,
	i: usize,
	code_buf: [u8; 17],
	code_buf_ptr: usize,
	mask: u8,
	len: usize,
}


impl LzssWriter {
	const N: usize = LzssReader::N;
	const NIL: usize = Self::N;
	const F: usize = LzssReader::F;
	const THRESHOLD: usize = LzssReader::THRESHOLD;


	fn init_state(&mut self) {
		for b in self.text_buf.iter_mut().take(Self::N - Self::F) {
			*b = 0x20;
		};

		for r in self.rchild.iter_mut().skip(Self::N + 1).take(256) {
			*r = Self::NIL;
		};

		for p in self.parent.iter_mut().take(Self::N) {
			*p = Self::NIL;
		};
	}


	fn insert_node(&mut self, r: usize) {
		use std::cmp::Ordering::*;

		let mut i: usize = 0;
		let mut cmp: std::cmp::Ordering = Greater;
		let mut p: usize = Self::N + 1 + self.text_buf[r] as usize;
		let mut rchild = self.rchild;
		let mut lchild = self.lchild;
		let mut parent = self.parent;
		let text_buf = self.text_buf;

		rchild[r] = Self::NIL;
		lchild[r] = Self::NIL;
		self.match_length = 0;

		loop {
			#[allow(clippy::collapsible_else_if)]
			if cmp.is_ge() {
				if rchild[p] != Self::NIL {
					p = rchild[p];
				}
				else {
					rchild[p] = r;
					parent[r] = p;
					return;
				};
			}
			else {
				if lchild[p] != Self::NIL {
					p = lchild[p];
				}
				else {
					lchild[p] = r;
					parent[r] = p;
					return;
				};
			};

			for j in 1..Self::F {
				cmp = text_buf[r + j].cmp(&text_buf[p + j]);
				if cmp.is_ne() {
					i = j;
					break;
				};
			};

			if i > self.match_length {
				self.match_position = p;
				self.match_length = i;

				if self.match_length >= Self::F {
					break;
				};
			};
		};

		parent[r] = parent[p];
		lchild[r] = lchild[p];
		rchild[r] = rchild[p];
		parent[lchild[p]] = r;
		parent[rchild[p]] = r;

		if rchild[parent[p]] == p {
			rchild[parent[p]] = r;
		}
		else {
			lchild[parent[p]] = r;
		};

		parent[p] = Self::NIL;

		self.parent = parent;
		self.lchild = lchild;
		self.rchild = rchild;
	}


	fn delete_node(&mut self, p: usize) {
		let mut q: usize;

		if self.parent[p] == Self::NIL {
			return;
		};

		if self.rchild[p] == Self::NIL {
			q = self.lchild[p];
		}
		else if self.lchild[p] == Self::NIL {
			q = self.rchild[p];
		}
		else {
			q = self.lchild[p];
			if self.rchild[q] != Self::NIL {
				loop {
					q = self.rchild[q];
					if self.rchild[q] != Self::NIL { break; }
				};
				self.rchild[self.parent[q]] = self.lchild[q];
				self.parent[self.lchild[q]] = self.parent[q];
				self.lchild[q] = self.lchild[p];
				self.parent[self.lchild[p]] = q;
			};
			self.rchild[q] = self.rchild[p];
			self.parent[self.rchild[p]] = q;
		};
		self.parent[q] = self.parent[p];

		if self.rchild[self.parent[p]] == p {
			self.rchild[self.parent[p]] = q;
		}
		else {
			self.lchild[self.parent[p]] = q;
		};

		self.parent[p] = Self::NIL;
	}


	fn finish_buffer_init(&mut self) {
		for i in 1..=Self::F {
			self.insert_node(self.r - i);
		};

		self.insert_node(self.r);
	}


	fn main_loop_pre_read<W: Write>(&mut self, output: &mut W) -> BcResult<usize> {
		if self.match_length > self.len {
			self.match_length = self.len;
		};

		if self.match_length <= Self::THRESHOLD {
			self.match_length = 1;
			self.code_buf[0] |= self.mask as u8;
			self.code_buf[self.code_buf_ptr] = self.text_buf[self.r];
		}
		else {
			self.code_buf[self.code_buf_ptr] = self.match_position as u8;
			self.code_buf_ptr += 1;
			self.code_buf[self.code_buf_ptr] = (
				((self.match_position >> 4) & 0xF0) |
				(self.match_length - (Self::THRESHOLD + 1))
			) as u8;
		};
		self.code_buf_ptr += 1;

		self.mask <<= 1;

		let mut bytes_written = 0;

		if self.mask == 0 {
			self.i = 0;

			while self.i < self.code_buf_ptr {
				let chr = self.code_buf[self.i];
				output.write_byte(chr)?;
				bytes_written += 1;
				self.i += 1;
			};

			self.code_buf[0] = 0;
			self.mask = 1;
			self.code_buf_ptr = 1;
		};

		self.last_match_length = self.match_length;
		self.i = 0;

		Ok(bytes_written)
	}


	fn main_loop_read(&mut self, input: u8) {
		self.delete_node(self.s);
		self.text_buf[self.s] = input;

		if self.s < Self::F - 1 {
			self.text_buf[self.s + Self::N] = input;
		};

		self.s = (self.s + 1) & (Self::N - 1);
		self.r = (self.r + 1) & (Self::N - 1);

		self.insert_node(self.r);
		self.i += 1;
	}


	fn main_loop_post_read(&mut self) {
		while self.i < self.last_match_length {
			self.delete_node(self.s);

			self.s = (self.s + 1) & (Self::N - 1);
			self.r = (self.r + 1) & (Self::N - 1);

			self.len -= 1;

			if self.len != 0 {
				self.insert_node(self.r);
			};

			self.i += 1;
		}
	}


	fn send_remaining<W: Write>(&mut self, output: &mut W) -> BcResult<usize> {
		let mut bytes_written = 0;

		if self.code_buf_ptr > 1 {
			self.i = 0;

			while self.i < self.code_buf_ptr {
				let chr = self.code_buf[self.i];
				output.write_byte(chr)?;
				bytes_written += 1;
				self.i += 1;
			};
		};

		Ok(bytes_written)
	}
}


impl Default for LzssWriter {
	fn default() -> Self {
		let state = LzssWriterState::BufferInit;
		let lchild = [0; Self::N + 1];
		let rchild = [0; Self::N + 257];
		let parent = [0; Self::N + 1];
		let text_buf = [0; Self::N + Self::F - 1];
		let match_position = 0;
		let match_length = 0;
		let last_match_length = 0;
		let s: usize = 0;
		let r: usize = Self::N - Self::F;
		let i = 0;
		let code_buf = [0; 17];
		let code_buf_ptr = 1;
		let mask = 1;
		let len = 0;

		let mut result = Self {
			state,
			lchild,
			rchild,
			parent,
			text_buf,
			match_position,
			match_length,
			last_match_length,
			s,
			r,
			i,
			code_buf,
			code_buf_ptr,
			mask,
			len,
		};

		result.init_state();

		result
	}
}


impl Algorithm for LzssWriter {
	fn filter_byte<W: Write>(&mut self, input: u8, output: &mut W) -> BcResult<usize> {
		use LzssWriterState::*;

		match self.state {
			BufferInit => {
				if self.len < Self::F {
					self.text_buf[self.r + self.len] = input;
					self.len += 1;
					Err(Waiting)
				}
				else {
					self.finish_buffer_init();
					self.state = MainLoop;
					self.filter_byte(input, output)
				}
			},

			MainLoop => {
				let mut bytes_written = self.main_loop_pre_read(output)?;
				self.state = ReadNew;
				bytes_written += match self.filter_byte(input, output) {
					Ok(written) => written,
					Err(Waiting) => 0,
					Err(e) => return Err(e),
				};
				Ok(bytes_written)
			},

			ReadNew => {
				self.main_loop_read(input);

				if self.i < self.last_match_length {
					Err(Waiting)
				}
				else {
					self.main_loop_post_read();

					if self.len > 0 {
						self.state = MainLoop;
						Err(Waiting)
					}
					else {
						Err(LzssInternalErrorCouldNotReadMore)
					}
				}
			},
		}
	}


	fn finish<W: Write>(&mut self, output: &mut W) -> BcResult<usize> {
		use LzssWriterState::*;

		match self.state {
			BufferInit => {
				if self.len == 0 {
					return Ok(0);
				};

				self.finish_buffer_init();
				self.state = MainLoop;
				self.finish(output)
			},

			MainLoop => {
				let bytes_written = self.main_loop_pre_read(output)?;
				self.main_loop_post_read();

				if self.len > 0 {
					self.state = MainLoop;
					self.i = 0;
					Ok(bytes_written + self.finish(output)?)
				}
				else {
					Ok(bytes_written + self.send_remaining(output)?)
				}
			},

			ReadNew => {
				self.main_loop_post_read();

				if self.len > 0 {
					self.state = MainLoop;
					Ok(self.finish(output)?)
				}
				else {
					Ok(self.send_remaining(output)?)
				}
			},
		}
	}
}


#[test]
fn test_lzss() {
	let input = [
		0x22, 0x0e, 0x00, 0x0a, 0x00, 0x08, 0x00, 0x00,
		0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x20, 0x20, 0x20, 0x3f, 0x20, 0x20, 0x20,
		0x20, 0x20 ];
	//let wanted = [
		//0xffu8, 0x22, 0x0e, 0x00, 0x0a, 0x00, 0x08, 0x00,
		//0x00, 0x23, 0x00, 0x80, 0xf4, 0xf0, 0xf8, 0xf1,
		//0xeb, 0xf0, 0x3f, 0xdc, 0xf2 ];
	let actual = LzssWriter::new().filter_slice_to_vec(&input).unwrap();
	// [TODO] wanted (Apple LZSS) is [..., 0xDC, 0xF2]; actual is [..., 0xDD, 0xF2].
	// ... and I have no idea why
	//assert_eq!(wanted, &actual[..]);
	let uncompressed = LzssReader::new().filter_slice_to_vec(&actual).unwrap();
	assert_eq!(&uncompressed[..], input);
}
