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
            bool isMale = true;
            bool isTall = true;

            if (isMale && isTall){
                Console.WriteLine("yessss");
            }
            else if (isMale || isTall){
                Console.WriteLine("dk");
            }
            else {
                Console.WriteLine("Nope");
            }
        }
        
    }
}
