package main

import "fmt"

type gasEngine struct {
	mpg     uint8
	gallons uint8
}
type electricEngine struct {
	mpkwh uint8
	kwh   uint8
}

func (e gasEngine) milesLeft() uint8 {
	return e.gallons * e.mpg
}
func (e electricEngine) milesLeft() uint8 {
	return e.kwh * e.mpkwh
}

type engine interface {
	milesLeft() uint8
}

func canMakeIt(e engine, miles uint8) bool {
	return miles <= e.milesLeft()
}
func main() {
	var myEngine gasEngine = gasEngine{mpg: 25, gallons: 15}
	myEngine.mpg = 28
	fmt.Println(myEngine)
	fmt.Printf("total miles left %v \n", myEngine.milesLeft())
	fmt.Printf("chat, can i make it? %v", canMakeIt(myEngine, 10))
}

/*
type gasEngine struct {
	mpg     uint8
	gallons uint8
	owner   owner
}
this will have like
{
mpg: smthn
gallons: smthn
owner: {
name: "Alex"
}
}


type gasEngine struct {
	mpg     uint8
	gallons uint8
	owner
}
this will have like{
mpg: smthn
gallons: smthn
name: "Alex"
}
*/

/*
var myEngine2 = struct {
		mpg     uint8
		gallons uint8
	}{25, 15}
	// the myEngine2 is a way to create a temp struct but that specific structure cant really be reused per say yk?
*/
