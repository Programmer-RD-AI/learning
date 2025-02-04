// fetch api

fetch("./todos/lugii.json")
  .then((data) => {
    if (data.status === 200) {
      return data.json();
    }
  })
  .then((data) => {
    console.log(data);
  })
  .catch((err) => {
    console.log(err);
  });

// An error is only raised when there is some sort of network error not normal mistake url
