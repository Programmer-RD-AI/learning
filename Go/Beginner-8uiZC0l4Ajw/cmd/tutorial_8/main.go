package main

import (
	"fmt"
	"sync"
	"time"
)

// RWMutex is specifically for Reading and Writing, Mutex is just general purpose, and this provides functions like RLock(), etc...
var m = sync.RWMutex{}
var wg = sync.WaitGroup{}
var dbData = []string{"id1", "id2", "id3", "id4", "id5"}
var results = []string{}

func main() {
	t0 := time.Now()
	for i := 0; i < len(dbData); i++ {
		wg.Add(1)
		go dbCall(i)
	}
	wg.Wait()
	fmt.Printf("total execution time: %v", time.Since(t0))
}
func dbCall(i int) {
	var delay float32 = 2000
	time.Sleep(time.Duration(delay) * time.Millisecond)
	var result string = dbData[i]
	save(result)
	log()
	wg.Done()
}

func save(result string) {
	m.Lock()
	results = append(results, result)
	m.Unlock()
}

func log() {
	m.RLock()
	fmt.Println("the results from the database is: ", results)
	m.RUnlock()
}
