#![no_main]
use libfuzzer_sys::fuzz_target;
use bohemia_compression::{Algorithm, LzssReader, LzssWriter};


#[allow(unused_variables)]
fn decompress_lzss_slice(input: &[u8]) -> Vec<u8> {
	// See https://opensource.apple.com/source/xnu/xnu-201/libsa/mkext.c.auto.html
	const N: usize = 4096;
	const F: usize = 16;
	const THRESHOLD: usize = 2;

	let mut dst_buf: Vec<u8> = Vec::new();
	let mut src: usize = 0;
	let srcend = input.len();

	let mut text_buf = [0x20u8; N + F - 1];

	let mut i: libc::c_int;
	let mut j: libc::c_int;
	let mut k: libc::c_int;
	let mut r: libc::c_int = (N - F) as libc::c_int;
	let mut c: libc::c_int;
	let mut flags: libc::c_uint = 0;

	loop {
		flags >>= 1;

		if (flags & 0x100) == 0 {
			if src < srcend { c = input[src].into(); src += 1; } else { break; };

			flags = libc::c_uint::from_le_bytes(c.to_le_bytes()) | 0xFF00;
		}

		if flags & 1 != 0 {
			if src < srcend { c = input[src].into(); src += 1; } else { break; };
			dst_buf.push((c & 0xFF) as u8);
			text_buf[r as usize] = (c & 0xFF) as u8;
			r += 1;
			r &= (N - 1) as libc::c_int;
		}
		else {
			if src < srcend { i = input[src].into(); src += 1; } else { break; };
			if src < srcend { j = input[src].into(); src += 1; } else { break; };
			i |= (j & 0xF0) << 4;
			j = (j & 0x0F) + THRESHOLD as libc::c_int;

			let pr = r;

			for k in 0..=j {
				c = text_buf[(pr - i + k) as usize & (N - 1)] as libc::c_int;
				dst_buf.push(c as u8);
				text_buf[r as usize] = c as u8;
				r += 1;
				r &= (N - 1) as libc::c_int;
			}
		}
	}

	dst_buf
}


fuzz_target!(|data: &[u8]| {
	let compressed_data = LzssWriter::new().filter_slice_to_vec(data).unwrap();
	let uncompressed_data = LzssReader::new().filter_slice_to_vec(&compressed_data[..]).unwrap();
	assert_eq!(uncompressed_data, data);
	let uncompressed_data_apple = decompress_lzss_slice(&compressed_data[..]);
	assert_eq!(uncompressed_data_apple, data);

	let bc_data = LzssReader::new().filter_slice_to_vec(data).unwrap();
	let apple_data = decompress_lzss_slice(data);
	assert_eq!(bc_data, apple_data);
});
