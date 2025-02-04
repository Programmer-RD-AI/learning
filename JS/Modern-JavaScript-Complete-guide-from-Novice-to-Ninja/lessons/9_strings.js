// strings
console.log("hello, world!");
let email = "go2ranuga@gmail.com";
console.log(email);
// string concatenation

let firstName = "Brandon";
let lastName = "Sanderson";
let fullName = firstName + " " + lastName;
console.log(fullName);

// getting characteristic
console.log(fullName[0]);

// string length
console.log(fullName.length);

// string methods
fullName = fullName.replace(" ", "-");
console.log(fullName);
fullName = fullName.toUpperCase();
console.log(fullName);

let result = fullName.toLowerCase();
console.log(result, fullName);

let index = email.indexOf("@");
console.log(index, email, email[index]);
