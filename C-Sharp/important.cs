// Imports
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace important
{
    class Program
    {
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
    }
}
