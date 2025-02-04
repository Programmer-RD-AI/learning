class User {
  constructor(username, email) {
    // set up properties in an object
    this.username = username;
    this.email = email;
  }
  login() {
    console.log(`${this.username} just logged in`);
  } // arrow functions dont use this object instance.

  logout() {
    console.log(`${this.username} logged out`);
  }
}

const userOne = new User("mario", "mario@gmail.com");
