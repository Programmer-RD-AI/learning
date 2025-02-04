export {};
let message = "Hello World";
console.log(message);

let x = 10;
const y = 20;
x = 4;

let sum;
const title = "Hello World!";

let isBeginner: boolean = true;
let total: number = 0;
let name: string = "Ranuga";

let sentence: string = `My name is ${name}
I am a beginner in TypeScript`;

console.log(sentence);
let n: null = null;
let u: undefined = undefined;
let isNew: boolean = null;
let myName: string = undefined;

let list1: number[] = [1, 2, 3];
let list2: Array<number> = [1, 2, 3];

let [person1]: [string, number] = ["Chris", 22];

enum Color {
  Red = 5,
  Green,
  Blue,
}

let c: Color = Color.Green;
console.log(c);

let randomValue: any = 10;
randomValue = true;
randomValue = "Ranuga";

let myVariable: unknown = 10;

function hasName(obj: any): obj is { name: string } {
  return !!obj && typeof obj === "object" && "name" in obj;
}
if (hasName(myVariable)) {
  console.log(myVariable.name);
}
// (myVariable as string).toUpperCase();

let a;
a = 10;
a = true;

let b = 10;
let multiType: number | boolean;

// function add(num1: number, num2?: number): number {
//   if (num2) {
//     return num1 + num2;
//   }
//   return num1;
// }
function add(num1: number, num2: number = 10): number {
  if (num2) {
    return num1 + num2;
  }
  return num1;
}
// interface Person {
//   firstName: string;
//   lastName: string;
// }
interface Person {
  firstName: string;
  lastName?: string;
}
function fullName(person: Person) {
  console.log(`${person.firstName} ${person.lastName}`);
  return person.firstName + " " + person.lastName;
}

let p = {
  firstName: "Ranuga",
  lastName: "Disansa",
};
fullName(p);

class Employee {
  public employeeName: string;
  constructor(name: string) {
    this.employeeName = name;
  }
  private  greet() {
    console.log(`Good Morning ${this.employeeName}`);
  }
}

class Manager extends Employee {
  constructor(managerName: string) {
    super(managerName);
  }

  delegateWork() {
    console.log(`Manager delegating tasks ${this.employeeName}`);
  }
}

let m1 = new Manager("Bruce");
m1.delegateWork();
