// Compound Data Types
// arrays, tuples, slices and strings (slice string)

// arrays
fn main() {
    let numbers: [i32; 5] = [1, 2, 3, 4, 5];
    println!("Array: {:?}", numbers);
    // let mix = [1, 2, "apple", true];
    // println!("Mixed Array: {:?}", mix);
    let fruits: [&str; 5] = ["apple", "banana", "cherry", "date", "elderberry"];
    println!("Fruits: {:?}", fruits);
    println!("1 Fruit: {}", fruits[0]);
    println!("2 Fruit: {}", fruits[1]);

    // Tuples
    let human: (String, i32, bool) = ("Alice".to_string(), 30, false);
    println!("Human Tuple: {:?}", human);

    let my_mix_tuple = ("Kratos", 23, true, [1, 2, 3, 4, 5]);
    println!("My Mix Tuple: {:?}", my_mix_tuple);

    // Slices: [1,2,3,4,5]
    let number_slices: &[i32] = &[1, 2, 3, 4, 5];
    println!("Number Slices: {:?}", number_slices);

    let animal_slices: &[&str] = &["cat", "dog", "fish"];
    println!("Animal Slices: {:?}", animal_slices);

    let book_slices: &[&String] = &[
        &"The Catcher in the Rye".to_string(),
        &"To Kill a Mockingbird".to_string(),
        &"1984".to_string(),
    ];
    println!("Book Slices: {:?}", book_slices);

    // Strings Vs String Slices (&str)
    // Strings [growable, mutable, owned string type]
    let mut stone_cold: String = String::from("Hello");
    println!("Stone Cold Says: {}", stone_cold);
    stone_cold.push_str("World!");
    println!("Stone Cold Says: {}", stone_cold);

    // B - &str (string slice)
    let string: String = String::from("Hello, World!");
    let slice: &str = &string[0..5];
    println!("String Slice: {}", slice);
}
