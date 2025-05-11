enum Color{
    Red,
    Green,
    Blue,
}

fn color_to_hex(c: Color) -> &'static str{
    match c {
        Color::Red => "#FF0000",
        Color::Green => "#00FF00",
        Color::Blue => "#0000FF",
    }
}

fn main(){
    println!("Red in hex is {}", color_to_hex(Color::Red));
}
