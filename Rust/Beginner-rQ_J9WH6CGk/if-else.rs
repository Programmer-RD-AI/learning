// If Else [ If Expression ] [ Else Expression ]
#![allow(warnings)]
fn main() {
    // let age: u16 = 18;
    // if age >= 18 {
    //     println!("You can drive a car!");
    // } else {
    //     println!("You can not drive a car!");
    // }

    // Multiple Conditions with else if:
    let number: i32 = 6;
    if number % 4 == 0 {
        println!("number is divisible by 4")
    } else if number % 3 == 0 {
        println!("number is divisible by 3")
    } else if number % 2 == 0 {
        println!("number is divisible by 2")
    } else {
        println!("number is not divisible by 4, 3, or 2")
    }

    // Using if in a let statement:
    let condition: bool = true;
    let number: i32 = if condition { 5 } else { 6 };
    println!("number is: {}", number);
}
