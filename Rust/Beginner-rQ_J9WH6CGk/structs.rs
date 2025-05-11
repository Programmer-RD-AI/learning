#![allow(warnings)]

// Structs
// Structs are used to name and package related values similar to tuples

fn main() {
    // tuple
    let rect: (i32, i32) = (200, 500);

    // Struct
    struct Book {
        title: String,
        author: String,
        pages: u32,
        available: bool,
    }

    struct User {
        active: bool,
        username: String,
        email: String,
        sign_in_count: u64,
    }

    let mut user1: User = User {
        active: true,
        username: String::from("user1"),
        email: String::from("go2ranuga@gmail.com"),
        sign_in_count: 1,
    };

    user1.email = String::from("ranuga.20231264@iit.ac.lk");
    println!("User email: {}", user1.email);

    fn build_user(username: String, email: String) -> User {
        User {
            active: true,
            username,
            email,
            sign_in_count: 1,
        }
    }

    // Create instances from other instances
    let user2: User = User {
        email: String::from("go2ranuga@gmail.com"),
        ..user1
    };

    // Tuple Structs
    struct Color(i32, i32, i32);
    struct Point(i32, i32, i32);

    let black: Color = Color(0, 0, 0);
    let white: Color = Color(255, 255, 255);

    // Unit Like Struct
    struct AlwaysEqual;
    let subject = AlwaysEqual;
}
