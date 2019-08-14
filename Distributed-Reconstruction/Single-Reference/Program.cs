using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            Single_Reference.GPUDeconvolution.GreedyCD2.TestRowMajor();
            //DebugMethods.DebugSimulatedGreedy2();
            DebugMethods.DebugILGPU();

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

    }
}
