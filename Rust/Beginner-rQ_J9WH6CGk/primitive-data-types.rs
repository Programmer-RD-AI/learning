// Primitive data types
// int, float, bool, char

// Integer
// i8, i16, i32, i64, i128, isize: Signed integers
// u8, u16, u32, u64, u128, usize: Unisgned integers
// Rust has signed (+ and -) and unsigned (only +) integers
fn main() {
    let x: i32 = -42;
    let y: u64 = 100;
    println!("Signed Integer: {}", x);
    println!("Unsigned Integer: {}", y);
    // diff bet i32 (32 bits) and i64 (64 bits)
    // range: i32 - 2147483647
    // range: i64 - 9223372036854775807

    // Floats [Floating Point Types]
    // f32, f64
    let pi: f64 = 3.14;
    println!("Pi: {}", pi);

    // Boolean Values: true, false
    let is_snowing: bool = true;
    println!("Is it snowing? {}", is_snowing);

    // Character Type - char
    let letter: char = 'A';
    println!("Letter: {}", letter);
}
