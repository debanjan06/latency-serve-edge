use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

pub struct ZeroCopyTensorReader {
    mmap: Mmap,
    pub total_elements: usize,
}

impl ZeroCopyTensorReader {
    /// Maps a raw binary tensor file straight into the virtual address space.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        if !file_size.is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Binary payload size must be a multiple of 4 bytes for float32 precision views",
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        let total_elements = file_size / 4;

        Ok(Self {
            mmap,
            total_elements,
        })
    }

    /// Borrows a clean window slice straight from the mapped pointer without copying data.
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const f32, self.total_elements) }
    }
}
