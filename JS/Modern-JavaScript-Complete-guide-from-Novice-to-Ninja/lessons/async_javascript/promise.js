// promise example
const getSomething = () => {
  return new Promise((resolve, reject) => {
    // do something like calling an API
    // resolve("some data");
    reject("some error");
  });
};
getSomething().then(
  (data) => {
    console.log(data);
  },
  (err) => {
    console.log(err);
  }
); // first one if resolve, second one if reject

getSomething()
  .then((data) => {
    console.log(data);
  })
  .catch((err) => {
    console.log(err);
  });

// both the above are the same
