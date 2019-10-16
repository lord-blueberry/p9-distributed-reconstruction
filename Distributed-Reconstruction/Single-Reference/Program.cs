using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            Experiments.PSFMask.Run();
            //Experiments.GPUSpeed.Run();
            //Experiments.PSFSize.RunSpeed();
            Experiments.PSFSize.RunPSFSize();



            //Experiments.PSFSize.DebugConvergence();
            //Experiments.PSFSize.DebugConvergence2();


            //DebugMethods.DebugSimulatedPCDM();
            //DebugMethods.DebugSimulatedApprox();
            //DebugMethods.DebugILGPU();

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
            //DebugMethods.DebugdWStack();
        }

    }
}
