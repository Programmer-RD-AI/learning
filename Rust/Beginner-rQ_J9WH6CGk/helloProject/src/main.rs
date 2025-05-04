// Compound Data tYPES
// arrays, tuples, slices and strings (slice string)

// arrays
fn main() {
    let numbers: [i32; 5] = [1, 2, 3, 4, 5];
    println!("Array: {:?}", numbers);
    // let mix = [1, 2, "apple", true];
    // println!("Mixed Array: {:?}", mix);
    let fruits: [&str; 5] = ["apple", "banana", "cherry", "date", "elderberry"];
}
