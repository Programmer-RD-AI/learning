// fn main(){
// let number_list: Vec<i32> = vec![1, 2, 3, 4, 5];
// let mut largest: i32 = number_list[0];
// for &number in &number_list {
//         if number > largest {
//             largest = number;
//         }
//     }
//
//     println!("The largest number is: {}", largest);
// }

fn get_largest<T: PartialOrd + Copy>(number_list: Vec<T>) -> T {
    let mut largest: T = number_list[0];

    for &number in &number_list {
        if number > largest {
            largest = number;
        }
    }

    largest
}

fn main() {
    let number_list: Vec<i32> = vec![1, 2, 3, 4, 5];
    let largest = get_largest(number_list);
    println!("The largest number is: {}", largest);

    let float_list: Vec<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5];
    let largest_float = get_largest(float_list);
    println!("The largest float is: {}", largest_float);
}
