// Functions
// Entry Point
// an function / variables should be written in snake case
// snake case: hello_word
// kebab case: hello-word
fn main() {
    hello_word();
    tell_height(123);
    human_id("name", 12, 0.5);
    let _X = {
        let price = 5;
        let qty = 10;
        price * qty
    };
    print!("X: {}", _X);
    // add(4, 6);
    let y: i32 = add(4, 6);
    println!("Y: {}", y);

    // Calling the BMI function
    let weight = 70.0;
    let height = 1.82;
    let bmi = calculate_bmi(weight, height);
    println!("Your BMI is: {:.2}", bmi);
}

// Hoisiting - can call function anywhere in your code
fn hello_word() {
    println!("Hello, word!");
}

// you can insert input values

fn tell_height(height: i32) {
    println!("Your height is: {}", height);
}

// you can insert more than one value

fn human_id(name: &str, age: u32, height: f32) {
    println!(
        "My name is: {}, I am {} years old, and my height is {}",
        name, age, height
    );
}

// functions returning values
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

// Expressions and Statements
// Expressions: anything that returns a value
// Statements: anything that does not return a value, Almost all statements in rust end with ;

// Expressions:
// 5
// true and false
// add(3,4)
// if condition {value1} else {value2}
// ({code})
// let _X = {
//     let price = 5;
//     let qty = 10;
//     price * qty
// };

// Statements:
// Almost all statements in Rust end with ;
// let y = let x = 10;
// 1 Variable declarations: let x = 5;
// 2 Function definitions: fn foo() {}
// 3 Control flow statements: if condition {/* code */} else { /* code */ }, while condition {/* code */}, for item in iterable {/* code */}

// Final Example
// BMI = weight(kg) / height(m)*2

fn calculate_bmi(weight_kg: f64, height_m: f64) -> f64 {
    return weight_kg / (height_m * height_m);
}
