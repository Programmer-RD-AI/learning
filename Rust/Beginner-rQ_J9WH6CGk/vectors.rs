// Vector

fn main() {
    let mut v: Vec<i32> = Vec::new();
    let _the_vec: Vec<i32> = vec![1,2,3];
    
    let mut _the_numbers_vec: Vec<i32> = Vec::new();
    _the_numbers_vec.push(5);
    _the_numbers_vec.push(6);
    _the_numbers_vec.push(7);
    _the_numbers_vec.push(8);
    _the_numbers_vec.push(9);
    _the_numbers_vec.push(10);
    println!("{:?}", _the_numbers_vec);

    let _v = vec![1,2,3,4,5];
    let third: &i32 = &_v[2]; // Direct Indexing
    println!("Third Element: {third}");

    let third = _v.get(2);
    match third{
        Some(third) => println!("The third element is {third}"),
        None => println!("No elemtn")
    }
}

