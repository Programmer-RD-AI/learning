let ninjas = ["Shawn", "Ryu", "Chun Li"];
// ninjas[1] = 'ken';
// console.log(ninjas, ninjas[1]);

// let ages = [20, 25, 30, 35];
// console.log(ages[2]);

// let random = ["shaun", "crystal", 30, 20];
// console.log(random);

// console.log(ninjas.length);
// ninjas.length = 0;
// ninjas.splice(0, ninjas.length);
// console.log(ninjas);

// let results = ninjas.join(","); "Shawn,Ryu,Chun Li"
// let results = ninjas.indexOf("Ryu")
// let results = ninjas.concat(["ranuga", "disansa"]);
let results = ninjas.push("ranuga"); // return the new length or the index of the new position
results = ninjas.pop();
console.log(results);
