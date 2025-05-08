// Shadowing
// Shadowing is not the same as s marking a variable as mutable.

fn main() {
    let x = 5; // 5
    let x = x + 1; // 6
    {
        let x = x * 2;
        println!("x: {x}"); // 12
    }
    println!("x: {}", x); // 6
}
