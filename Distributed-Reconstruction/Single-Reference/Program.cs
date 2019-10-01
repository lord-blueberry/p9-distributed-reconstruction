using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            //Single_Reference.Experiments.PSFSize.Run();

            //Experiments.PSFSize.DebugConvergence();
            
            //Deconvolution.ToyImplementations.RandomBlockCD2.RunToy();
            //DebugMethods.DebugSimulatedGreedy2();
            //DebugMethods.DebugSimulatedPCDM();
            
            DebugMethods.DebugILGPU();

            Console.WriteLine("hello");
            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

    }
}
