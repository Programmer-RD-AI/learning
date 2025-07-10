package main

import (
	"fmt"
	"unicode/utf8"
)

// uint = only positive integer

func main(){
	var intNum uint = 32767;
	intNum = intNum + 1;
	fmt.Println(intNum);
	// var floatNum float64;
	// fmt.Println(floatNum)
	var floatNum float32 = 10.1;
	var intNum32 int32 = 2;
	var result float32 = floatNum + float32(intNum32)
	fmt.Println(result)

	var intNum1 int = 3
	var intNum2 int = 2
	fmt.Println(intNum1/intNum2)
	fmt.Println(intNum1%intNum2)

	var myString string = "Hello World"
	fmt.Println(myString)

	var testString = `
	Multi Line Strings type shi
	`
	fmt.Println(testString)

  /* 
	so here we have to do this since when we call the len() function in go it provides us the number of bytes 
	(and in scenario where we get weird aaa characters then it does not exactly work per say yk?)
	*/
	fmt.Println(utf8.RuneCountInString("weird sigma symbol type shi"))

	var myRune rune = 'a';
	fmt.Println(myRune)

	var myBoolean bool = false
	fmt.Println(myBoolean)

	var intNum3 int
	fmt.Println(intNum3)

	// var myVar = "text"
	// same as:
	myVar := "text"
	fmt.Println(myVar)

	var var1, var2 = 1, 2
	fmt.Println(var1, var2)

	const myConst string = "const value"
	fmt.Println(myConst)

	const pi float32 = 3.1415
}
