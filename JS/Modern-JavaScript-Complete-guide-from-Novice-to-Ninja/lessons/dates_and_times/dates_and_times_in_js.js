const now = new Date();
console.log(now, typeof now);
// yrs, months, day, time
console.log(
  now.getFullYear(),
  now.getMonth(),
  now.getDate(),
  now.getMinutes(),
  now.getDay()
);
// timestamps
console.log(now.getTime()); // miliseconds since january 1st
// date strings
console.log(
  now.toDateString(),
  now.toTimeString(),
  now.toLocaleString().split(", ")
);
