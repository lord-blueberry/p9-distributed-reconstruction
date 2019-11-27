using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;

namespace SingleMachineRuns
{
    class UserQueries
    {
        public static void FirstQuery()
        {
            Console.WriteLine("Select dataset");
            Console.WriteLine("\t 0 -- Simulated two point");
            Console.WriteLine("\t 1 -- Subset of LMC dataset");
            var dataset = ReadInt(1, "standard (0)");

            Console.WriteLine("Select algorithm");
            Console.WriteLine("\t 0 -- Serial CD");
            Console.WriteLine("\t 1 -- Serial CD GPU");
            Console.WriteLine("\t 2 -- Parallel CD");
            var algo = ReadInt(1, "(Default 0):");
        }

        public static void QueryConfiguration(GriddingConstants c)
        {
            Console.WriteLine("Set image pixel size (Default 3072)");
            Console.WriteLine("Set cell size in arcsecs (Default 1.5)");
        }

        public static void QueryLambaAlpha()
        {
            Console.WriteLine("Set Lambda parameter of ElasticNet (larger than 0, default 0.5)");
            Console.WriteLine("Set Alpha parameter of ElasticNet (between 0 and 1, default 0.1)");
        }

        private static double ReadDouble()
        {
            return 0.0;
        }

        private static int ReadInt(int max, string question, int defaultValue = 0)
        {
            bool valid = false;
            int output = 0;
            do
            {
                Console.Write(question);
                var ret = Console.ReadLine();
                try
                {
                    if (ret == "")
                    {
                        output = defaultValue;
                    } else
                    {
                        output = Int32.Parse(ret);
                        if (output > 0 && output <= max)
                            valid = true;
                        else
                            Console.WriteLine("invalid integer. Should be between 0 and " + max);
                    }
                } 
                catch(Exception e)
                {
                    Console.WriteLine("invalid, not an integer");
                }
            } while (!valid);
            Console.WriteLine("read " + output);
             
            return output;
        }
    }
}
