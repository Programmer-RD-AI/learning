// logical operators - OR || and AND &&
const password = "pssword12345";

if (password.length >= 12 && password.includes("@")) {
} else if (
  password.length >= 8 ||
  (password.includes("@") && password.length >= 5)
) {
  console.log("that password is strong enough 💀");
} else {
  console.log("that password is not strong enough...");
}
