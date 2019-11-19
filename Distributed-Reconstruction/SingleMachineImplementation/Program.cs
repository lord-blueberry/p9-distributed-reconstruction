using System;
using Single_Reference;

using System.IO;
using System.Numerics;
using static System.Math;
using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;
using static SingleMachineRuns.Experiments.DataLoading;

namespace SingleMachineRuns
{
    class Program
    {
        static void Main(string[] args)
        {
            //Experiments.GPUSpeed.Run();
            //Experiments.PSFSize.RunSpeedLarge();

            Experiments.PCDMComparison.Run();

            //ImageGeneration.ApproxImages.Generator.Generate();
            //Experiments.PSFSize.RunApproximationMethods();
            //Experiments.ApproxParameters.Run();

            //ImageGeneration.Generator.GenerateAll();
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
