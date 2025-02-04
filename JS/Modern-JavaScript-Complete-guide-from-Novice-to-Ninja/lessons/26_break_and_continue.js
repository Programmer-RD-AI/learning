const scores = [50, 25, 0, 30, 100, 20, 10];
for (let i = 0; i < scores.length; i++) {
  console.log(`your score : ${scores[i]}`);
  if (scores[i] == 0) {
    continue;
  }
  if (scores[i] === 100) {
    console.log("congratz u beat the highest score!");
    break;
  }
}
