using System;

using System.Threading.Tasks;
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
            //RunningMethod.StartSimulatedReconstruction();
            //RunningMethod.StartLMCReconstruction();

            //Experiments.GPUSpeed.Run();
            //Experiments.PSFSize.RunApproximationMethods();
            //Experiments.PSFSize.RunApproximationMethods();
            //Experiments.ApproxParameters.Run();

            //Experiments.PCDMComparison.CalcESOs();
            //Experiments.PCDMComparison.Run();


            FigureGeneration.Figures.GenerateAll();
        }
    }
}
