# Learning C#

- [C# Starting Intro](https://www.youtube.com/watch?v=ravLFzIguCM) [and](https://www.youtube.com/watch?v=6LvQyQvaVZs)

- [C# Basics](https://www.youtube.com/watch?v=GhQdlIFylQ8)

- [C# Intermideate](https://www.youtube.com/watch?v=qOruiBrXlAw)

- [C# Advanced](https://codewithmosh.com/p/the-ultimate-csharp-mastery-series)

# REST API
`dotnet new sln -o [Project Name]`
`cd [Project Name]`
`dotnet new classlib -o [Project Name].Contracts`
`dotnet new webapi -o [Project Name]`  
`dotnet sln add ./[Project Name].Contracts/ ./[Project Name]/`

# C#

Status: In progress

- Declare a variables
    
    ```csharp
    **string characterName = "Nani";**
    ```
    
- print
    
    ```csharp
    $"There once was a man named {characterName}"
    ```
    
- read a int
    
    ```csharp
    int random_string =  Convert.ToInt32(Console.ReadLine());
    ```
    
- list
    
    ```csharp
    List<int> grades = new List<int>();
    List<int> grades = new List<int>() {5,10};
    grades.Add(5);
    Console.WriteLine(grades.count); // 1
    ```
    
- array
    
    ```csharp
    int[] array = {0, 1, 2};
    string[] friends = {"test","grgr","grgr"};
    friends[0] = "test";
    Console.WriteLine($"{friends[1]}");
    ```
    
- functions / methods
    
    ```csharp
    static void Main(string[] args)
    {
    	string characterName = "Nani";
    	Console.WriteLine($"There once was a man named {characterName}");
    	int random_string =  Convert.ToInt32(Console.ReadLine());
    	int[] array = {0, 1, 2};
    	string[] friends = {"test","grgr","grgr"};
    	friends[0] = "test";
    	Console.WriteLine($"{friends[1]}");
    	Console.WriteLine($"{input()}");
    }
    public static string input(){
    	string whatever = Console.ReadLine();
    	return whatever;
    }
    ```
    
- Switch
    
    ```csharp
    switch (param){
    	case 0:
    		test = "1";
    		break;
    	case 1:
    		test = "2";
    		break;
    	case 2:
    		test = "3";
    		break;
    	case 3:
    		test = "4";
    		break
    }
    ```
    
- While
    
    ```csharp
    int index = 0;
    while (index >= 5){
    	Console.WriteLine(index);
    	index++;
    }
    ```
    
    - Do While
        
        ```csharp
        int index = 0;
        do{
        	Console.WriteLine(index);
        	index++;
        }while (index >= 5)
        ```
        
- For While
    
    ```csharp
    for (int i = 0; i<=5; i++){
    	Console.WriteLine(i);
    }
    ```
    
- Grid Arrays
    
    ```csharp
    int [,] numberGrid = { // more , the more dimensions
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                };
    Console.WriteLine(numberGrid[1,2]); // row, column (starts with 0)
    Console.ReadLine();
    ```
    
- Try and Execpt
    
    ```csharp
    try{}
    catch{}
    finally{}
    // else:
    // catch (Exception e)
    // catch (DivideByZeroException e)
    // expect SimpleError:
    // execpt Exception as e:
    ```
    
- Classes
    
    ```csharp
    class Book
        {
            public string Title { get; set; }
    
            public string Author { get; set; }
    
            public string Publisher { get; set; }
    
            public int Year { get; set; }
    
            public int Pages { get; set; }
    
            public string Isbn { get; set; }
        }
    Book book1 = new Book();
    ```
    
    - Contructors
        
        ```csharp
        class Book
        {
        	public string Title;
        
          public string Author;
        
        	public string Publisher;
        
        	public int Year;
        
        	public int Pages;
        
        	public string Isbn;
        
        	public Book(string aTitle, string aAuthor, string aPublisher, int aYear, int aPages,string aIsbn)
                {
                    Title = aTitle;
                    Author = aAuthor;
                    Publisher = aPublisher;
                    Year = aYear;
                    Pages = aPages;
                    Isbn = aIsbn;
                }
            }
        // the public Book() is like __init__() in python
        ```
        
    - Object Functions
        
        ```csharp
        public string cool_out_string_lol()
        {
        return "cool";
        }
        ```
        
    - Requirments Paramaeter
        
        ```csharp
        public int year
        {
        	get // Console.WriteLine(object.year)
        		{
        			return year;
            }
        	set // object.year = 2849
        		{
        			if (value == 2000){
        				year = value;
               Console.WriteLine(year);
        	    }
        	    else
        	      {
        						Console.WriteLine("Invalid year");
                }
        	}
        }
        ```
        
    - Satic
        - Its a thing that is common for all objects
        - Its essentially like when one changes all changes
        
        ```csharp
        Book.songCount // this is common to all objects
        harry_potter.name // this is a specific thing to a object
        ```
        
    - Inheritence
        
        ```csharp
        class BadProgram : Program
        ```
        
        - need to add “virtual” in functions which will be overrided

# Theory

- Static
    - When some function is static it can be accesed with out the need of creating a instance of the class.
    - So Math.add() is a static method
    - but new m = new Math(); m.add() is not a static method
- Old List
    
    ```csharp
    List<int> array = new List<int> {5,10};
    ```
