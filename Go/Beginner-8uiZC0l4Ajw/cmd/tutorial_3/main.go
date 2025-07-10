package main

import (
	"errors"
	"fmt"
)

func main() {
	var printValue string = "Hello Word"
	printMe(printValue)

	var numerator int = 11
	var denominator int = 2
	var result, remainder, err = intDivision(numerator, denominator)
	// if err!= nil {
	// 	fmt.Println(err.Error())
	// } else if remainder == 0{
	// 	fmt.Printf("the result of the integer division is %v", result)
	// } else {
	// fmt.Printf("The result of the integer divison is %v and the remainder is %v", result, remainder)
	// }
	switch {
	case err != nil:
		fmt.Printf(err.Error())
	case remainder == 0:
		fmt.Printf("the result of the intger divison is %v", result)
	default:
		fmt.Printf("the result of the integer division is %v with remainder %v", result, remainder)
	}

	switch remainder {
	case 0:
		fmt.Printf("the division was exact")
	case 1, 2:
		fmt.Printf("the division was close")
	default:
		fmt.Printf("the division was not close")
	}
}

func printMe(printValue string) {
	fmt.Println(printValue)
}

func intDivision(numerator int, denominator int) (int, int, error) {
	var err error
	if denominator == 0 {
		err = errors.New("cannot divide by zero")
		return 0, 0, err
	}
	var result int = numerator / denominator
	var remainder int = numerator % denominator
	return result, remainder, err
}

// if 1==1 && or || 2==2
