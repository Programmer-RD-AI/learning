// Ownership, Borrowing and References

// Ownership
// ---------
// C, C++ -> Memory Management Control Issue
// Garbage Collector solved this issue, but created a new issue -> Slow Performance: [stopping/resuming the program]

// What is Ownership?
// Every value has a single owner [every variable has one value, and it is its sole owner]

// Ownership Rules
// 1: Each value in Rust has a variable that's its owner.
// 2: There can be only one owner at a time.
// 3: When the owner goes out of scope, the value will be dropped.

// Example: Each value in rust has a variable that's its owner.
fn ex1() {
    let s1 = String::from("RUST");
    let len = calculate_length(&s1); // Passing a reference to the function
    println!("Length of {} is {}", s1, len);
}

// fn calculate_length(s: &String) -> usize {
//     return s.len();
// }
// 2: There can be only one owner at a time.

fn ex2() {
    let s1 = String::from("RUST");
    let s2 = s1; // s1 is moved to s2, s1 is no longer valid
}

// 3: When the owner goes out of scope, the value will be dropped.

fn main() {
    let s1 = String::from("RUST");
    let len = calculate_length(&s1); // Passing a reference to the function
    println!("Length of {} is {}", s1, len);
}
fn printList(s: &String) {
    println!("{}", &s1);
}
fn calculate_length(s: &String) -> usize {
    return s.len();
}
