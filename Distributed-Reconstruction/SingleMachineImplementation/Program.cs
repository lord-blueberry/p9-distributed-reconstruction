using System;

namespace SingleMachineRuns
{
    class Program
    {
        static void Main(string[] args)
        {
            //Experiments.GPUSpeed.Run();
            //Experiments.PSFSize.RunSpeedLarge();
            //Experiments.PSFSize.RunPSFSize();



            //Experiments.PSFMask.Run();
            Experiments.ApproxParameters.Run();
            //Experiments.ApproxParameters.ActiveSetDebug();

            //Single_Reference.DebugMethods.DebugILGPU();
            //Single_Reference.DebugMethods.DebugSimulatedApprox();

            //Experiments.PSFSize.DebugConvergence();
            //Experiments.PSFSize.DebugConvergence2();



            //

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
            //DebugMethods.DebugdWStack();
        }
    }
}
