fn main() {
    // Loop keyword
    // loop {
    //     // Print a message
    //     println!("Hello, world!");

    //     // Break the loop
    //     // break;
    // }

    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            // Break the loop and return the value
            break counter * 2;
        }
    };
    // Print the result
    println!("The result is: {}", result);

    // Loop Labels to Disambiguate Between Multiple Loops
    
}
