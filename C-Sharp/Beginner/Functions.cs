using System;
using System.Collections.Generic;

namespace
Function // organization structure
{
    class Function // class
    {
        static void Function_Main(string[] args) // function // dotnet run -- {some text}
        {
            // Program myProgram = new Program();
            // myProgram.Print("This is being called as a non static method");
            Program.Print("This is being called as a static method");
        }

        static void Print(string what_to_say)
        {
            Console.WriteLine (what_to_say);
        }
    }
}
