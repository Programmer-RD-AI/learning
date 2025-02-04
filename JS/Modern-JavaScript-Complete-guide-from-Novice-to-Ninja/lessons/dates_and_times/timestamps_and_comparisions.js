const before = new Date("February 1 2023 7:30:59");
const now = new Date();
// console.log(now.getTime(), before.getTime());
// const now_g = new Date(now.getTime());
// console.log(now_g, now);
const diff = now.getTime() - before.getTime();
console.log(diff);
const mins = Math.round(diff / 1000 / 60);
const hours = Math.round(mins / 60);
const days = Math.round(hours / 24);
console.log(mins, hours, days);

console.log(`the blog was written ${days} ago`);

const timestamp = 167593847990;
const timestamp_Date = new Date(timestamp);
console.log(timestamp_Date);
