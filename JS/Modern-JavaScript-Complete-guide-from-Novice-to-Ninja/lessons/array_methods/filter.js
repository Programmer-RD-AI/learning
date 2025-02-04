const scores = [10, 30, 15, 25, 50, 40, 5];
// const filteredScores = scores.filter((score) => {
//   return score > 10;
// });
// console.log(scores, filteredScores);
const users = [
  { name: "shaun", premium: true },
  { name: "yoshi", premium: false },
  { name: "mario", premium: false },
  { name: "chun-li", premium: true },
];
const premium_users = users.filter((user) => {
  return user.premium;
});
console.log(premium_users);
