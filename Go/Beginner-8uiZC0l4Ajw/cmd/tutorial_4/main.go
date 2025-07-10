package main

import "fmt"

func main() {
	var intArr [3]int32
	intArr[1] = 123
	fmt.Println(intArr[0])

	// Continuous Memory, So that each of the elements are next to each other so that we know the locations yk?
	fmt.Println(&intArr[0])

	// var intArry [3]int32 = [3]int32{1, 2, 3}
	intArry := [...]int32{1, 2, 3}
	fmt.Println(intArry)

	var intSlice []int32 = []int32{4, 5, 6}
	fmt.Println(intSlice)
	intSlice = append(intSlice, 7)
	fmt.Println(intSlice)

	var intSlice2 []int32 = []int32{8, 9}
	intSlice = append(intSlice, intSlice2...)
	fmt.Println(intSlice)

	var intSlice3 []int32 = make([]int32, 3, 8)
	fmt.Println(intSlice3)

	var myMap map[string]uint8 = make(map[string]uint8)
	fmt.Println(myMap)

	var myMap2 = map[string]uint8{"Adam": 23, "Sarah": 45}
	fmt.Println(myMap2["Adam"])
	var age, ageExist = myMap2["Jason"]
	// delete(myMap2, "Adam")
	if ageExist {
		fmt.Printf("the age is %v", age)
	} else {
		fmt.Println("invalid key brh1")
	}

	for name, age := range myMap2 {
		fmt.Printf("Name %v, age %v", name, age)
	}

	for i, v := range intArr {
		fmt.Printf("Index %v, Value %v", i, v)
	}
	var i int = 0
	// for i > 10 {
	// 	fmt.Println(i)
	// 	i += 1
	// }
	for {
		if i > 10 {
			break
		}
		fmt.Println(i)
		i += 1
	}
	for i := 0; i < 10; i++ {
		fmt.Println(i)
	}
}
