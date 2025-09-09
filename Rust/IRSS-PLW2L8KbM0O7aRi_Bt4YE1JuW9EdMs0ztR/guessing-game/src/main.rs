fn main() {
    let actual: &str = "blue";
    println!("Welcome to the guessing game!");
    println!("I have chosen a color: red, green or blue can you guess which?");
    println!("Enter your guess: red, green or blue");
    let input = std::io::stdin()
        .lines()
        .next()
        .expect("No input was read")
        .expect("There was an error when reading the input");
    println!("The color i chose was {actual}")
}
