const ul = document.querySelector(".people");
const people = [
  "mario",
  "luigi",
  "ryu",
  "shaun",
  "chun-li",
  "ranuga",
  "disansa",
];
let html = "";
people.forEach((element) => {
  html += `<li style="color:purple">${element}</li>`;
});
ul.innerHTML = html;
