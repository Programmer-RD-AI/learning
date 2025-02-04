const userOne = {
  username: "ryu",
  email: "ryu@thenetninja.co.uk",
  login() {
    console.log("the user logged in");
  },
  logout() {
    console.log("the user logged out");
  },
};
console.log(userOne.email, userOne.username);
userOne.login();
