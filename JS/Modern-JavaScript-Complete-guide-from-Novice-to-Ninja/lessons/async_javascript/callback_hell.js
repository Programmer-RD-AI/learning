const getTodos = (callback, location) => {
  const request = new XMLHttpRequest();
  request.addEventListener("readystatechange", () => {
    if (request.readyState === 4 && request.status === 200) {
      const data = JSON.parse(request.responseText);
      callback(undefined, data);
    } else if (request.readyState === 4) {
      callback("could not fetch data", undefined);
    }
  });
  // request.open("GET", "https://jsonplaceholder.typicode.com/todos/");
  request.open("GET", location);
  request.send();
};
console.log(1);
console.log(2);

getTodos((err, data) => {
  console.log(data, err);
  getTodos((err, data) => {
    console.log(data, err);
    getTodos((err, data) => {
      console.log(data, err);
    }, "./todos/shaun.json");
  }, "./todos/mario.json");
}, "./todos/lugii.json");

console.log(3);
console.log(4);
