/*
Channels: Hold Data, Thread Safe, Listen for Data
*/

package main

import "fmt"

func main() {
	var c = make(chan int, 5)
	go process(c)
	// fmt.Println(<-c)
	for i := range c {
		fmt.Println(i)
	}
}

func process(c chan int) {
	defer close(c)
	// c <- 123
	for i := 0; i < 5; i++ {
		c <- i
	}
	close(c)
}
