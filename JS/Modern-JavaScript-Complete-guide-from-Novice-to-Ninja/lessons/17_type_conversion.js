let score = "100";
console.log(typeof score);
score = Number(score);
console.log(score + 1);
// Double(score);
// String(score);
// Boolean(score);
console.log(typeof score);

let result = "hello";
console.log(Number(result)); // NaN

result = String(69);
console.log(result);
console.log(typeof result);

result = Boolean(69); // 0 is False, all others are positive
console.log(result, typeof result);

result = Boolean("0"); // all strings are true except if it is empty...
console.log(result, typeof result);
