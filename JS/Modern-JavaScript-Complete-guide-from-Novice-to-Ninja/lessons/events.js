// const button = document.querySelector("button");
// button.addEventListener("click", () => {
//   console.log("You clicked me !");
// });
const items = document.querySelectorAll("li");
items.forEach((element) => {
  element.addEventListener("click", (e) => {
    // element.remove();
    e.target.style.textDecoration = "line-through";
  });
});
