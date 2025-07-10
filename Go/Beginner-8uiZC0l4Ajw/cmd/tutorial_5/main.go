package main

import (
	"fmt"
	"strings"
)

func main() {
	var myString = "résumé"
	var myRunes = []rune("résumé")
	var indexed = myString[0]
	fmt.Println(myRunes)
	fmt.Println(indexed)
	fmt.Println(myString)

	var strSlice = []string{"s", "u", "b", "s", "c", "r", "i", "b", "e"}
	var strBuilder strings.Builder
	for i := range strSlice {
		strBuilder.WriteString(strSlice[i])
	}
	var catStr = strBuilder.String()
	fmt.Printf("%v", catStr)
}

/*
so in fmt.Printf()

we can use %T to get the type of a specific value

var value int = 1;
fmt.Printf("%v %T", value, value)
*/

/*
so in the string setup its a bit complicated ig but not that bad, here is how it goes

golang uses utf 8 which is a dynamic byte setup where a specific character might take up 8 or 16 or continued no. of bits, we dont really know
so in turn it is really hard for us to manage it right, so because of that when we index for example we get the uint no. of that specific byte, but the thing is 1 character can have more than 1 byte so its a bit of a tricky situation, you can go through the for loop and get it as well but still too complicated.

the easiest way it seems is to pretty much just use runes since they will auto calculate the uint of a character since its represented individually
*/
