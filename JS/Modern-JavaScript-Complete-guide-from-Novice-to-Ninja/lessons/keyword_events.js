const form = document.querySelector(".signup-form");
const feedback = document.querySelector(".feedback");
const usernamepattern = /^[a-zA-Z]{6,12}$/;
form.addEventListener("submit", (e) => {
  e.preventDefault(); // prevents the default action of a event (in this is to refresh the page)
  const username = form.username.value;

  if (usernamepattern.test(username)) {
    // feedback
    feedback.textContent = "The user name fills out the criteria";
    feedback.style.color = "green";
  } else {
    feedback.textContent =
      "The user name dont fills out the criteria (only letters and between 6 and 12 characters";
    feedback.style.color = "red";
  }
});

// live feedback
form.username.addEventListener("keyup", (e) => {
  if (usernamepattern.test(e.target.value)) {
    e.target.setAttribute("class", "success");
  } else {
    e.target.setAttribute("class", "error");
  }
});
