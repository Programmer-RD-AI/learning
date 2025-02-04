const express = require("express");
const app = express();
app.set("view engine", "ejs");
app.use(logger);

app.use(express.static("public/"));
app.use(express.urlencoded({ extended: true }));
app.get("/", logger, (req, res) => {
  console.log("Here");
  //   res.sendStatus(404);
  //   res.status(500).send("500 Error");
  //   res.status(500).json({ message: "Error" });
  //   res.download("./package-lock.json");
  //   res.json({ test: true });
  res.render("index", { txt: "World" });
});
const userRouter = require("./routes/users");
// const postRouter = require("./routes/posts");
app.use("/users", userRouter);

function logger(req, res, next) {
  console.log(req.originalUrl);
  next();
}
// app.use("/posts", postRouter);
app.listen(3000);
