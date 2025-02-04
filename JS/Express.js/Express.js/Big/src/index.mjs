import express from "express";
const app = express();
app.use(express.json());
const PORT = process.env.PORT || 3000;
const mockUsers = [
  { id: 1, username: "anson", displayName: "ANson" },
  { id: 2, username: "joe", displayName: "joez" },
  { id: 3, username: "mamam", displayName: "MAMAZ" },
];
app.get("/", (req, res) => {
  res.status(201).send({ msg: "hello" });
});
app.get("/api/users", (req, res) => {
  console.log(req.query);
  const {
    query: { filter, value },
  } = req;
  if (filter && value)
    return res.send(mockUsers.filter((user) => user[filter].includes(value)));
  return res.send(mockUsers);
});
app.post("/api/users", (req, res) => {
  const { body } = req;
  const user = {
    id: mockUsers[mockUsers.length - 1].id + 1,
    ...body,
  };
  mockUsers.push(user);
  console.log(mockUsers);
  return res.send(200);
});
app.get("/api/users/:id", (req, res) => {
  const parsedID = parseInt(req.params.id);
  if (isNaN(parsedID)) return response.status(404).send({ msg: "Bad Request" });
  const findUser = mockUsers.find((user) => user.id === parsedID);
  if (!findUser) return response.status(404).send({ msg: "Not Found" });
  res.send(findUser);
});
app.get("/api/products", (req, res) => {
  res.send([
    { id: 1, name: "product1", price: 100 },
    { id: 2, name: "product2", price: 200 },
    { id: 3, name: "product3", price: 300 },
    { id: 4, name: "product4", price: 400 },
    { id: 5, name: "product5", price: 500 },
    { id: 6, name: "product6", price: 600 },
    { id: 7, name: "product7", price: 700 },
    { id: 8, name: "product8", price: 800 },
    { id: 9, name: "product9", price: 900 },
    { id: 10, name: "product10", price: 1000 },
  ]);
});
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
