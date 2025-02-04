const age = 30;
// const age = 50; // i cant redeclare the same variable
if (true) {
  const age = 40; // local scope or it is only availalbe inside here..
  const name = "shaun";
  console.log(`inside code block : ${age} ${name}`);
  if (true) {
    const age = 50;
    console.log(name, age);
  }
  console.log(age);
}
console.log(`outside code block : ${age}`);
