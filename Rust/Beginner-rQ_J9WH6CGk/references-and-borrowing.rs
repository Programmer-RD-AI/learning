// References and Borrowing
// Safety and Performance
// Borrowing and References are powerful concepts

// Understanding References
// References allow you to borrow data without taking ownership
// Immutable references allow you to read data without modifying it
// Mutable references allow you to modify data, but only one mutable reference is allowed at a time
// Create Reference by adding & before the variable name

// fn main() {
//     let mut _x: i32 = 5;
//     let _r: &mut i32 = &mut _x;
//     *_r += 1;
//     println!("The value of x is: {}", _x);
// }

// Demonstration on one mutable reference or many immutable references

fn main() {
    let mut account = BankAccount {
        owner: String::from("Alice"),
        balance: 150.55,
    };
    // Immutable borrow to check the balance
    account.check_balance();
    // Mutable borrow to withdraw money
    account.withdraw(50.0);
    // Immutable borrow to check the balance
    account.check_balance();
}

struct BankAccount {
    owner: String,
    balance: f64,
}

impl BankAccount {
    fn withdraw(&mut self, amount: f64) {
        println!(
            "Withdrawing {} from account owned by {}",
            amount, self.owner
        );
        self.balance -= amount;
    }

    fn check_balance(&self) {
        println!(
            "Account owned by {} has a balance of {}",
            self.owner, self.balance
        );
    }
}
