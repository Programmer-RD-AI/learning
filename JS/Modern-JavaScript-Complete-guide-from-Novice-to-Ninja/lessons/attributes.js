const link = document.querySelector("a");
console.log(link.getAttribute("href"));
link.setAttribute("href", "https://thenetninja.com");
link.innerText = "TheNet Ninja Website";

const msg = document.querySelector("p");
console.log(msg.getAttribute("class"));
msg.setAttribute("class", "success");
msg.setAttribute("style", "color:green");
