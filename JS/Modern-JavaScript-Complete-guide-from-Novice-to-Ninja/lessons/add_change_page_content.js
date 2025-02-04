const para = document.querySelector("p");
// para.textContent = "Updated text";

const paras = document.querySelectorAll("p");
// paras.forEach((para, index) => {
//   para.textContent += ` Updated this one too... ${index} before (${para.textContent})`;
// });

const content = document.querySelector(".content");
// console.log(content.innerHTML);
// content.innerHTML = `<h2>
//   THIS IS THE NEW H2
// </h2>`;

const people = ["mario", "luigi", "yoshi"];
people.forEach((person, index) => {
  content.innerHTML += `<p>${person}</p>`;
});
