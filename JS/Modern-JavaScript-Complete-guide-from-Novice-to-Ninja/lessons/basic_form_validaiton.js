const form = document.querySelector(".signup-form");
const feedback = document.querySelector(".feedback");
form.addEventListener("submit", (e) => {
  e.preventDefault(); // prevents the default action of a event (in this is to refresh the page)
  const username = form.username.value;
  const usernamepattern = /^[a-zA-Z]{6,12}$/;
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
