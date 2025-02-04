const request = new XMLHttpRequest();
request.addEventListener("readystatechange", (e) => {
  //   console.log(request, request.readyState, request.responseText);
  if (request.readyState === 4 && request.status === 200) {
    console.log(request.responseText);
  } else if (request.readyState === 4) {
    console.log("Could not fetch the data");
  }
  //   console.log(request.status);
  // 0 -> UNSENT
  // 1 -> OPENED
  // 2 -> HEADER RECEIVED
  // 3 -> Loading
  // 4 -> COMPLETED
});
request.open("GET", "https://jsonplaceholder.typicode.com/todos/");
request.send();
