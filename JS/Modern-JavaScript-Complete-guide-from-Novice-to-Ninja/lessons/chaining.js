const products = [
  { name: "gold start", price: 30 },
  { name: "green shell", price: 10 },
  { name: "red shell", price: 40 },
  { name: "banana skin", price: 5 },
  { name: "mushroom", price: 50 },
];
const promos = filtered
  .map((value) => {
    return `the ${value.name} is ${value.price / 2}`;
  })
  .filter((value) => {
    return value.price > 20;
  });
console.log(promos, filtered);
