// callbacks
const myFunc = (callbackFunc) => {
  let value = 50;
  callbackFunc(value);
};
myFunc((value) => {
  console.log(value);
});
let people = ["mario", "luigi", "ryu", "shaun", "chun-li"];
// people.splice(0, 1);

const logPerson = (person, index) => {
  console.log(`${index} hellow ${person}`);
};
people.forEach(logPerson);
