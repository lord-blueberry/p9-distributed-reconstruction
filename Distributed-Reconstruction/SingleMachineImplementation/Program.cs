using System;
using Single_Reference;

namespace SingleMachineRuns
{
    class Program
    {
        static void Main(string[] args)
        {
            //Experiments.GPUSpeed.Run();
            //Experiments.PSFSize.RunSpeedLarge();
            //Experiments.PSFSize.RunPSFSize();

            var image = FitsIO.ReadImage("ApproxTest/xImage_4.fits");
            var x2 = ImageGeneration.Tools.LMC.CutCalibration(image);
            FitsIO.Write(x2, "n132.fits");
            


            //Experiments.PSFMask.Run();
            //Experiments.ApproxParameters.Run();
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
