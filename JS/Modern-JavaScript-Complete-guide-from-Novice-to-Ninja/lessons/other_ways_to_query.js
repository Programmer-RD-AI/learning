// get an element by ID
const title = document.getElementById("page-title");
console.log(title);
// get elements by their class name
const errors = document.getElementsByClassName("error"); // cant use forEach in the HTMLCollection type of array
console.log(errors);
// get elements by their tag name
const p_tags = document.getElementsByTagName("p");
console.log(p_tags);
