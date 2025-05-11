fn main(){
    // if let example
    let config_max: Option<u8> = Some(7);

    if let Some(max) = config_max{
        println!("The maximum is configured to be {}", max);
    }

    // while let example
    let mut numbers = vec![1,2,3];
    while let Some(n) = numbers.pop(){
        println!("Popped {}", n);
    }

    // let ... else example
    let data = Some("value");
    let inner = let Some(v) = data else {
        panic!("No value found");
    }
    println!("The value is {}", inner);
}
