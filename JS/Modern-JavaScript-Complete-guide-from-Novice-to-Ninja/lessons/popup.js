const button = document.querySelector("button");
const popup = document.querySelector(".popup-wrapper");
const popup_close = document.querySelector(".popup-close");
button.addEventListener("click", () => {
  popup.style.display = "block";
});
popup_close.addEventListener("click", () => {
  popup.style.display = "none";
});
popup.addEventListener("click", () => {
  popup.style.display = "none";
});
