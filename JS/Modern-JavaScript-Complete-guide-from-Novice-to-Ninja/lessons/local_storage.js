// store data
localStorage.setItem("name", "hrterhrter");
localStorage.setItem("age", "50");
// get data
let name = localStorage.getItem("name");
let age = localStorage.getItem("age");
console.log(name, age);
// update data
localStorage.setItem("name", "gsfgsfd");
localStorage.age = "40";

name = localStorage.getItem("name");
age = localStorage.getItem("age");
console.log(name, age);

// delete data from local Storage

// localStorage.removeItem("name");

localStorage.clear(); // clear the entire local storage
name = localStorage.getItem("name");
console.log(name);
