const getTodos = (callback, location) => {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.addEventListener("readystatechange", () => {
      if (request.readyState === 4 && request.status === 200) {
        const data = JSON.parse(request.responseText);
        resolve(data);
      } else if (request.readyState === 4) {
        reject("couldnt not get some data");
      }
    });
    request.open("GET", location);
    request.send();
  });
};

getTodos("./todos/lugii.json")
  .then((data) => {
    console.log("promise resolved: ", data);
    // getTodos("./todos/mario.json");
  })
  .catch((err) => {
    console.log("promise rejected: ", err);
  });
