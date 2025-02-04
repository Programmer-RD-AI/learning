# Data Types

Numbers - 1,2,3,4,5,6,100,3.14
String - 'hello world', "<mario@thenetninja.co.uk>"
Boolean - true / false
Null - Explicitly set a variable
Undefined - Variables that aren't defined yet
Object - Complex data structures (Arrays, Dates, Literals, etc...)
Symbol - Used with objects..

# Control Flow

- Use loops for cycling through data
- Use conditional statement for checking conditions

# Function

- A function is like a box that runs the code inside it and we can call it and the box runs and we can use the same code again and again... `showScores()`

# Objects

- Real life have properties and they also do...
- For example an phone
  - properties
    - color
    - size
    - model
  - things it can do (methods / functions)
    - ring
    - take a picture
    - play music

# Primitive vs Reference types

## Primitive

Stored in the stack, the space in there is limited, much more faster
The stacks variable values refer to the actual variable name such as 'hello' -> greeting and so on...s
When one variable is assigned to another variable the value is copied.

- numbers
- strings
- booleans
- null
- undefined
- symbols

## Reference

Stored in the heap, the space is much more but more slower
The objects are stored in the heap but the pointers or the thing that will let us know which variable is which is still stored in the stack, and in turn the program has to go to the stack and find the data in the `heap`.
When one variable is assigned to another variable the value is the same meaning the point is the same for both variables and if the value is changed for 1 it is changed for both

- all types of objects
  - object literals
  - arrays
  - functions
  - dates
  - all other objects

# Document Object Model (DOM)

It the heart of an web page manipulation
Created by the browser, can be accessed through `document` object
The following is the structure of an DOM:
html -> body -> h1
-> div
-> p
-> p
-> div
-> head -> title

# CSS Selectors

- ID = #id
- CSS Class = .class
- Tag with Class = tag.class
- - = all

# Form Events

- Capture data or information from a user
- Eg: a username or password
- "submit" event is when a user clicks the submit button
- "keyboard" events is when we can know if the user has pressed something...
- RegEx is used to make sure that the value entered by the user is useful..

# Async JavaScript

## What is Asynchronous JavaScript?

One of the most important part of the javascript.
Governs how we perform tasks which take some time to complete (Getting data from a database)
Code that can start something now and finish it later
Synchronous javascript is where each statement is executed in an order, ane after the other and also only 1 at one time.
Block code blocks the next statement from running.

## What are HTTP Requests?

Make HTTP requests to get data from another server. We make these requests to API endpoints.

# Object-Oriented Programming

## Classes

A blueprint for an object (describes how it should be constructed)
