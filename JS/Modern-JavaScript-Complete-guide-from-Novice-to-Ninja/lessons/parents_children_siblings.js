const article = document.querySelector("article");
// Array.from(article.children).forEach((element) => {
//   child.classList.add("article-element");
//   if (element.tagName.toLowerCase() == "p") {
//   }
// });
const title = document.querySelector("h2");
console.log(title.parentElement.parentElement);
console.log(title.parentElement);
console.log(title.nextElementSibling);
console.log(title.previousElementSibling);

// chaining
console.log(title.nextElementSibling.parentElement.children);
