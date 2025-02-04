const ul = document.querySelector("ul");
// ul.remove();
const button = document.querySelector("button");
button.addEventListener("click", () => {
  // ul.innerHTML += "<li>something new</li>";
  const li = document.createElement("li");
  li.textContent = "something new to do";
  // ul.append(li);
  // ul.insertBefore(li, ul.children[3]); Add a to a specific position
  ul.prepend(li);
});

// const items = document.querySelectorAll("li");
// items.forEach((element) => {
//   element.addEventListener("click", (e) => {
//     console.log(e);
//     e.stopPropagation();
//     e.target.remove();
//   });
// });

ul.addEventListener("click", (e) => {
  // console.log("event in UL", e);
  console.log(e.target);
  if (e.target.tagName === "LI") {
    e.target.remove();
  }
});
