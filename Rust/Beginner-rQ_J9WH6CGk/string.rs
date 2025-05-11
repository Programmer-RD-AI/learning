fn main() {
    let s = "whatever".to_string();
    let s = String::from("whatever");
    // Mutate the variable [push to it]
    let mut s = String::from("foo");
    s.push_str("bar");
    s.push("!");
    println!("s = {s}");

    let s1 = String::from("Hello, ");
    let s2 = String::from("world!");
    let s3 = s1 + &s2; // Note: s1 is moved here and can no longer be used, but s2 is not moved
    println!("s3 = {s3}");

    // Formatting Strings
    let salam = String::from("salam");
    let salut = String::from("salut");
    let full_message: String format!("{salam} {salut}");
}
