const getTodos = async () => {
  const response = await fetch("./todos/lugii.json");
  console.log(response);
  if (response.status !== 200) {
    throw new Error("gg");
  }
  const data = await response.json();
  return data;
};

getTodos()
  .then((data) => {
    console.log(data);
  })
  .catch((err) => {
    console.log(err.message);
  });
