const form = document.querySelector(".signup-form");
// const username = document.querySelector("#username");
form.addEventListener("submit", (e) => {
  e.preventDefault(); // prevents the default action of a event (in this is to refresh the page)
  // console.log(username.value);
  // username.value = "";
  console.log(form.username.value); // name also similar case with . notation, class dont work..!
  form.username.value = "";
});
