// Enums
// Versatile tool used to represent a type that can take on one of several possible variants.

fn main() {
    enum IpAddressKind {
        V4,
        V6,
    }
    let _four = IpAddressKind::V4;
    let _six = IpAddressKind::V6;

    fn route(_ip_kind: IpAddressKind) {}

    route(IpAddressKind::V4);

    // Using structs
    // struct IpAddr {
    //     kind: IpAddressKind,
    //     address: String,
    // }

    // Using Enums
    enum IpAddr {
        V4(u8, u8, u8, u8),
        V6(String),
    }
    // let home: IpAddr = IpAddr::V4(String::from("127.0.0.1"));
    // let loopback: IpAddr = IpAddr::V6(String::from("::1"));

    // Enhanced Enums
    let home: IpAddr = IpAddr::V4(127, 0, 0, 1);
    let loopback: IpAddr = IpAddr::V6(String::from("::1"));
}

