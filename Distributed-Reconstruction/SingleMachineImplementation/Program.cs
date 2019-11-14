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

            /*
            var psfFull = FitsIO.ReadImage("psfFull_size.fits");
            var psfApprox = FitsIO.ReadImage("psfFull_approx.fits");

            var img = new float[psfFull.GetLength(0), psfFull.GetLength(1)];
            for (int i = 0; i < psfFull.GetLength(0); i++)
                for (int j = 0; j < psfFull.GetLength(1); j++)
                    img[i, j] = psfFull[i, j] - psfApprox[i, j];

            FitsIO.Write(img, "psf_diff.fits");
            Console.WriteLine(PSF.CalcMaxLipschitz(psfFull));
            Console.WriteLine(PSF.CalcMaxLipschitz(psfApprox));*/

            //Experiments.PSFSize.RunApproximationMethods();
            Experiments.ApproxParameters.Run();

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
