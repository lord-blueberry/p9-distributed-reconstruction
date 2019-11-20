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
            //Experiments.PSFSize.RunApproximationMethods();
            //Experiments.ApproxParameters.Run();
            Experiments.PCDMComparison.Run();

            //ImageGeneration.Generator.GenerateAll();
            //ImageGeneration.ApproxImages.Generator.Generate();

            //Single_Reference.DebugMethods.DebugILGPU();
            //Single_Reference.DebugMethods.DebugSimulatedApprox();


            //

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
            //DebugMethods.DebugdWStack();
        }
    }
}
