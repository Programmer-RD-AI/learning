const prices = [20, 10, 30, 25, 15, 40, 80, 5];
// const new_prices = prices.map((price) => {
//   return price * 1.25;
// });
// console.log(prices, new_prices);
const products = [
  { name: "gold start", price: 20 },
  { name: "mushroom", price: 40 },
  { name: "green shells", price: 30 },
  { name: "banana skin", price: 10 },
  { name: "red shells", price: 50 },
];
const saleProducts = products.map((product) => {
  if (product.price > 30) {
    // product.price = product.price / 2; // cant do this becz it edits the og list
    return { name: product.name, price: product.price / 2 };
  }
  return product;
});
console.log(saleProducts);
