// document.getElementsByTagName("h1")[0].innerText =
//   "This is the DOM Changing stuff.";

// const para = document.querySelector("div.error");
// console.log(para);

const paras = document.querySelectorAll("p");
paras.forEach((para) => {
  console.log(para);
});
// console.log(paras); // NodeList, not an array but some of the features
const errors = document.querySelectorAll(".error");
