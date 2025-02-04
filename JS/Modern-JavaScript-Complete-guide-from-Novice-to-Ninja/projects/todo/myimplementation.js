const add_todo = document.querySelector(".add");
const todos = document.querySelector("ul.todos");
const search = document.querySelector(".search");
add_todo.addEventListener("submit", (event) => {
  event.preventDefault();
  const li = document.createElement("li");
  li.innerHTML = `<li class="list-group-item d-flex justify-content-between align-items-center">
        <span>${event.target.add.value}</span>
        <i class="far fa-trash-alt delete"></i>
    </li>`;
  todos.appendChild(li);
  event.target.add.value = "";
});
todos.addEventListener("click", (event) => {
  if (event.target.tagName === "I") {
    todos.removeChild(event.target.offsetParent);
  }
});
search.addEventListener("submit", (event) => {
  event.preventDefault();

  Array.from(todos.children).some((todo, idx) => {
    console.log(
      todo.querySelector("span").innerText,
      typeof todo.querySelector("span").innerText,
      typeof event.target.search.value
    );
    if (
      todo.querySelector("span").innerText.includes(event.target.search.value)
    ) {
      todos.removeChild(todo);
      return true;
    }
    console.log(idx);
  });
  event.target.search.value = "";
});
