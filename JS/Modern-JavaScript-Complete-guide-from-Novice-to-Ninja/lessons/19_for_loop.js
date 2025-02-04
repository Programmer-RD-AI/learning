// for loops

// for (let i = 0; i < 10; i++) {
//   console.log(i);
//   // section of code that will iterate until the condition is true
// }

const names = ["shaun", "mario", "luigi"];
for (let i = 0; i < names.length; i++) {
  //   console.log(`The users name is : ${names[i]}`);
  let html = `<div>
        User Name : ${names[i]}
    </div>`;
  console.log(html);
}
