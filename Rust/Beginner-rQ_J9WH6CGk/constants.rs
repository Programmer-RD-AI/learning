// Constants

fn main() {
    println!("Hello, world!");
    // Can't use mut with constants
    let mut x = 5;
    const N: i32 = 10;
}

// You can declare a constant with a type annotation.
const PI: f64 = 3.1415926535897932385;
const THREE_HOURS_IN_SECONDS: u32 = 3 * 60 * 60;
