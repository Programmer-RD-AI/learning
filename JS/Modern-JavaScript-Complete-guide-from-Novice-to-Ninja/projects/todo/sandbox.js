const addForm = document.querySelector(".add");
const list = document.querySelector("ul");
const search = document.querySelector("input.ss");
const generate_template = (todo) => {
  const html = `<li class="list-group-item d-flex justify-content-between align-items-center">
            <span>${todo}</span>
            <i class="far fa-trash-alt delete"></i>
        </li>`;
  list.innerHTML += html;

  return html;
};
addForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const todo = addForm.add.value.trim(); // .trim() removes any blank space from the start and end
  if (todo.length) {
    //  > 0
    generate_template(todo);
    addForm.reset(); // reset all the input fields in an form
  } else {
    alert("There must be a value entered..!");
  }
});
// delete todos
list.addEventListener("click", (e) => {
  if (e.target.classList.contains("delete")) {
    e.target.parentElement.remove();
  }
});

// keyup event

const filterTodos = (term) => {
  Array.from(list.children)
    .filter((val) => {
      return !val.textContent.includes(term);
    })
    .forEach((todo) => {
      todo.classList.add("filtered");
    });
};
search.addEventListener("keyup", (e) => {
  e.name;
  const term = search.value.trim();
  filterTodos(term);
});
