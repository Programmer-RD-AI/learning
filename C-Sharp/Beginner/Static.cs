using System;
using System.Collections.Generic;

namespace
Static // organization structure
{
    class Static // class
    {
        static void Static_Main(string[] args) // function
        {
            Program myProgram = new Program();
            myProgram.Print();
            Program.Print();
        }

        static void Print()
        {
            Console.WriteLine("Hello World!");
            Console.WriteLine("Hello");
        }
    }
}
