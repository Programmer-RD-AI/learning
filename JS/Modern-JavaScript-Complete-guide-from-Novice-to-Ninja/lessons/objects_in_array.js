// objects in array
// const blogs = [
//   { title: "why mac and chesse is the best", likes: 30 },
//   { title: "10 things to make with chesse", likes: 50 },
// ];
let user = {
  name: "crystal",
  age: 30,
  email: "go2ranuga@gmail.com",
  location: "colombo",
  blogs: [
    { title: "why mac and chesse is the best", likes: 30 },
    { title: "10 things to make with chesse", likes: 50 },
  ],
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
      console.log(`${element.title} (${element.likes})`);
    });
    // `this` can be only accessed by normal functions and not arrow functions!
  },
};

user.logBlogs();
