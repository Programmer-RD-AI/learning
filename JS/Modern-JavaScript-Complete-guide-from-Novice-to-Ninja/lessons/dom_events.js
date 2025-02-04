const copy = document.querySelector(".copy-me");
copy.addEventListener("copy", () => {
  console.log("u copied me !!");
  //   copy.innerHTML =
  //     "<img src='https://static-00.iconduck.com/assets.00/middle-finger-black-emoji-350x512-61ls9nsv.png'/>";
});
const box = document.querySelector(".box");

box.addEventListener("mousemove", (e) => {
  // console.log(e.offsetX, e.offsetY);
  box.textContent = `${e.offsetX} ${e.offsetY}`; // inside the .box
});

document.addEventListener("wheel", (e) => {
  console.log(e.pageX, e.pageY); // inside the entire document
});
