package main

import "fmt"

//	func main() {
//		var p *int32 = new(int32)
//		var i int32
//		fmt.Printf("the value p points to is: %v", *p)
//		fmt.Printf("the value if i is: %v", i)
//		*p = 10 // so when we do new() have a memory location we point to right, we update the memory within that, the variable p still refers to the original memory location which has the memory location of the new() variable
//		p = &i  // now p and i both refer to the same spot in memory
//		*p = 1
//	}
func main() {
	var thing1 = [5]float64{1, 2, 3, 4, 5}
	fmt.Printf("\n the memory location of the thing1 array is: %p", &thing1)
	var result [5]float64 = square(&thing1)
	fmt.Printf("\n the result is: %v", result)
	fmt.Printf("\n the value of thing1 is: %v", thing1)
}

func square(thing2 *[5]float64) [5]float64 {
	for i := range thing2 {
		thing2[i] = thing2[i] * thing2[i]
	}
	return *thing2
}
