const speak = function (name = "luigi", time = "day") {
  console.log(`good ${time} ${name}!`);
};
speak("mario", "morning");
speak();
