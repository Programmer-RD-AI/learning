enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

// fn divide(numerator: f64, denominator: f64) -> Option<f64> {
//     if denominator == 0.0 {
//         Option::None
//     } else {
//         Option::Some(numerator / denominator)
//     }
// }

fn divide(numerator: f64, denominator: f64) -> Result<f64, String> {
    if denominator == 0.0 {
        Result::Err("Division by zero".to_string())
    } else {
        Result::Ok(numerator / denominator)
    }
}

fn main() {
    // let result = divide(10.0, 2.0);
    // match result {
    //     Option::Some(x) => println!("Result: {}", x),
    //     Option::None => println!("Error: Division by zero"),
    // }
    let result = divide(10.0, 2.0);
    match result {
        Result::Ok(x) => println!("Result: {}", x),
        Result::Err(err) => println!("Error: {}", err),
    }
}

