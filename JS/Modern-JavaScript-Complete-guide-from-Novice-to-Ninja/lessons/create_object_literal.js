// objects literals
let user = {
  name: "crystal",
  age: 30,
  email: "go2ranuga@gmail.com",
  location: "colombo",
  blogs: ["why mac and chesse is the best", "10 things to make with chesse"],
};
console.log(user);
console.log(user.name); // console.log(user["name"]);

user.age = 38;
console.log(user);
user["name"] = "chun-li";
console.log(typeof user);
// adding methods objects
let user = {
  name: "crystal",
  age: 30,
  email: "go2ranuga@gmail.com",
  location: "colombo",
  blogs: ["why mac and chesse is the best", "10 things to make with chesse"],
  login: () => {
    console.log("The user logged in..!");
  },
  logout: () => {
    console.log("The user logged out..!");
  },
  logBlogs() {
    // for (let i = 0; i < this.blogs.length; i++) {
    //   console.log(`This blog name is : ${this.blogs[i]}`)
    // }
    this.blogs.forEach((element, index) => {
      console.log(`${element} (${index})`);
    });
    // `this` can be only accessed by normal functions and not arrow functions!
  },
};
user.login();
user.logout();
user.logBlogs();
