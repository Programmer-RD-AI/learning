// const content = document.querySelector("p");
// // content.classList = "success";
// content.classList.add("error");
// content.classList.remove("error");
// content.classList.add("success");

const content = document.querySelectorAll("p");
content.forEach((element) => {
  if (element.textContent.includes("success")) {
    element.classList.add("success");
  }
  if (element.textContent.includes("error")) {
    element.classList.add("error");
  }
  // console.log(element.innerText.includes("success"));
});
// textContent > innerText
const title = document.querySelector(".title");
title.classList.toggle("test"); // adds if it isnt added already, removes if it already exists
