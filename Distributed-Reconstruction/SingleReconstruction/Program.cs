﻿using System;

using System.IO;
using System.Numerics;
using static System.Math;
using Core;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using static Core.Common;
using static SingleReconstruction.Experiments.DataLoading;

namespace SingleReconstruction
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