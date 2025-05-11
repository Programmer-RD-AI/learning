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
    let mut count = 0;
    'counting_up: loop {
        println!("count = {}", count);
        let mut remaining = 10;

        loop {
            println!("remaining = {}", remaining);
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }
        count += 1;
    }

    // While Loop
    let mut number = 3;
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }

    // Loop through a collection with for loop
    let a = [10, 20, 30, 40, 50];
    let b: [&str; 5] = ["a", "b", "c", "d", "e"];
    for element in a {
        println!("The value is: {}", element);
    }
}
