use std::fs::File;
use std::io::{self, Read};

// Recoverable with Result
fn read_username(path: &str) -> Result<String, io::Error> {
    let mut f = File::open(path)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// Unrecoverable with panic
fn index_example(){
    let v = vec![1, 2, 3];
    // This will panic if the index is out of bounds
    let _x = v[99];
}
