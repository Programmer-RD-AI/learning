// arrow functions

// const calcArea = function (radius) {
//   return 3.14 * radius ** 2;
// };
const calcArea = (radius) => 3.14 * radius ** 2;
const area = calcArea(5);
console.log(area);

// CONVERT

const greet = () => "hello, world";
const bill = (products, tax) => {
  let total = 0;
  for (let i = 0; i < products.length; i++) {
    total += products[i] + products[i] * tax;
  }
  return total;
};
