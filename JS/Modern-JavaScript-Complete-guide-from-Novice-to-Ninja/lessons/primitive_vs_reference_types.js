let scoreOne = 50;
let scoreTwo = scoreOne;

console.log(scoreOne, scoreTwo);

scoreOne = 65;
console.log(scoreOne, scoreTwo);

const userOne = {
  name: "ryu",
  age: 30,
};
const userTwo = userOne;
console.log(userOne, userTwo);
userOne.name = "gdf";
console.log(userOne, userTwo);
