const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
  res.send("user list");
});

router.get("/new", (req, res) => {
  res.render("users/new");
});

router.post("/", (req, res) => {
  console.log(req.body.firstName);
  res.send("user created");
});

router
  .route("/:id")
  .get((req, res) => {
    console.log(request.user);
    res.send("user profile", req.params.id);
  })
  .put((req, res) => {
    res.send("Update user profile", req.params.id);
  })
  .delete((req, res) => {
    res.send("Delete profile", req.params.id);
  });
const users = [{ name: "kyle" }, { name: "sALLY" }];
router.param("id", (req, res, next, id) => {
  req.user = users[id];
  next();
});
router.get("/:id");

module.exports = router;
