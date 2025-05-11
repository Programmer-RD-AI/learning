// Defining a custom enum

enum Shape{
    Circle(f64),
    Rectangle{width: f64, height: f64},
    Traingle(f64, f64, f64),
}

// Using the Option enum

fn divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None
    } else {
        Some(a / b)
    }
}

fn main(){
    let circle = Shape::Circle(5.0);
    let result = divide(10.0, 2.0);
    match result {
        Some(value) => println!("Result: {}", value),
        None => println!("Cannot divide by zero"),
    }
}
