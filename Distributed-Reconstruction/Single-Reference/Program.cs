using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            Deconvolution.RandomBlockCD.RunToy();
            //DebugMethods.DebugSimulatedGreedy2();
            DebugMethods.DebugSimulatedPCDM();
            
            //DebugMethods.DebugILGPU();

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

    }
}
