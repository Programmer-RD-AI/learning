struct Point<T, U> {
    x: T,
    y: U
}

fn main(){
    let point1 = Point { x: 10, y: 20 };
    let point2 = Point { x: 5.0, y: 10.0 };
    let point3 = Point { x: 5, y: 10.0 };
}
