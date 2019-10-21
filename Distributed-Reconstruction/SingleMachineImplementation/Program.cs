using System;

namespace SingleMachineRuns
{
    class Program
    {
        static void Main(string[] args)
        {
            //Experiments.GPUSpeed.Run();
            Experiments.PSFSize.RunPSFSize();
            //Experiments.PSFSize.RunSpeedSmall();
            //Experiments.PSFSize.RunSpeedLarge();


            //Experiments.PSFMask.Run();


            //Experiments.PSFSize.DebugConvergence();
            //Experiments.PSFSize.DebugConvergence2();


            Single_Reference.DebugMethods.DebugSimulatedPCDM();
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
