use my_app::utils::helper;
use my_app::models::Data;

fn main() {
    helper();
    let d = Data { id: 1, name: "Item".into() };
    println!("Loaded data: {} - {}", d.id, d.name);
}
