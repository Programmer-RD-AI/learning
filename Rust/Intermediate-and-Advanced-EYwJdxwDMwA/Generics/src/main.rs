fn last_element<T>(list: &[T]) -> &T{
    &list[list.len() - 1]
}
/*
fn main(){
    let integer_array: [i32; 4] = [23, 42, 61, 97];
    let float_array: [f64; 3] = [2.9, 7.3, 1.8];
    let string_slice_array = ["hello", "world", "rust"];
    let last_integer = last_element(&integer_array);
    let last_float = last_element(&float_array);
    let last_string = last_element(&string_slice_array);
    println!("Last integer: {}", last_integer);
    println!("Last float: {}", last_float);
    println!("Last string: {}", last_string);
}
*/

#[derive(Debug)]
struct Rainfalls<T>{
    place_1: T,
    place_2: T,
    place_3: T,
}

fn main(){
    let jan_23 = Rainfalls {
        place_1: 2.3,
        place_2: 3.4,
        place_3: 4.5,
    }
    println!("Rainfall in January 2023: {:?}", jan_23);    
}
