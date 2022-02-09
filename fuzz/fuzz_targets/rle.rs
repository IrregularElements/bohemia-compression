#![no_main]
use libfuzzer_sys::fuzz_target;

use std::io::Cursor;
use arbitrary::{
	Arbitrary,
	Unstructured,
	Result as ArbitraryResult,
};
use bohemia_compression::*;


#[derive(Debug)]
pub struct RleMinimumRunFuzzer {
	minimum_run: usize,
}


impl From<RleMinimumRunFuzzer> for usize {
	fn from(value: RleMinimumRunFuzzer) -> Self {
		value.minimum_run
	}
}


impl<'a> Arbitrary<'a> for RleMinimumRunFuzzer {
	fn arbitrary(input: &mut Unstructured) -> ArbitraryResult<Self> {
		Ok(Self { minimum_run: input.int_in_range(1..=0x79)? as usize })
	}
}



fuzz_target!(|input: (RleMinimumRunFuzzer, &[u8])| {
	let (minimum_run, data) = input;
	let mut input_cursor = Cursor::new(data);
	let mut compressed_data = Vec::<u8>::new();
	let mut writer = RleWriter::with_minimum_run(minimum_run.into());
	writer.filter_to_end(&mut input_cursor, &mut compressed_data).unwrap();

	let mut compressed_data_cursor = Cursor::new(compressed_data);
	let mut uncompressed_data = Vec::<u8>::new();
	let mut reader = RleReader::new();
	reader.filter_to_end(&mut compressed_data_cursor, &mut uncompressed_data).unwrap();
	assert_eq!(data, &uncompressed_data[..]);
});
