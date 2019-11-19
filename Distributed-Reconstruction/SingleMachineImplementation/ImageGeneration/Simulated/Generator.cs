using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Numerics;
using Single_Reference;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;

using static Single_Reference.Common;

namespace SingleMachineRuns.ImageGeneration.Simulated
{
    class Generator
    {
        public static void GeneratePSFs(string simulatedLocation, string outputFolder)
        {
            var data = MeasurementData.LoadSimulatedPoints(simulatedLocation);
            var c = MeasurementData.CreateSimulatedStandardParams(data.VisibilitiesCount);
            var metadata = Partitioner.CreatePartition(c, data.UVW, data.Frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, data.UVW, data.Flags, data.Frequencies);
            var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            Directory.CreateDirectory(outputFolder);

            var maskedPsf = Copy(psf);
            Tools.Mask(maskedPsf, 2);
            var reverseMasked = Copy(psf);
            Tools.ReverseMask(reverseMasked, 2);
            var psf2 = PSF.CalcPSFSquared(psf);
            var psf2Cut = PSF.CalcPSFSquared(maskedPsf); 

            Tools.WriteToMeltCSV(psf, Path.Combine(outputFolder, "psf.csv"));
            Tools.WriteToMeltCSV(maskedPsf, Path.Combine(outputFolder, "psfCut.csv"));
            Tools.WriteToMeltCSV(reverseMasked, Path.Combine(outputFolder, "psfReverseCut.csv"));
            Tools.WriteToMeltCSV(psf2, Path.Combine(outputFolder, "psfSquared.csv"));
            Tools.WriteToMeltCSV(psf2Cut, Path.Combine(outputFolder, "psfSquaredCut.csv"));

            var x = new float[c.GridSize, c.GridSize];
            x[10, 10] = 1.0f;

            var convKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, c.GridSize, c.GridSize));
            var corrKernel = PSF.CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, c.GridSize, c.GridSize));
            using (var convolver = new PaddedConvolver(convKernel, new Rectangle(0, 0, c.GridSize, c.GridSize)))
            using (var correlator = new PaddedConvolver(corrKernel, new Rectangle(0, 0, c.GridSize, c.GridSize)))
            {
                var zeroPadded = convolver.Convolve(x);
                var psf2Edge = correlator.Convolve(zeroPadded);
                Tools.WriteToMeltCSV(zeroPadded, Path.Combine(outputFolder, "psfZeroPadding.csv"));
                Tools.WriteToMeltCSV(psf2Edge, Path.Combine(outputFolder, "psfSquaredEdge.csv"));
            }
            convKernel = PSF.CalcPaddedFourierConvolution(psf, new Rectangle(0, 0, 0, 0));
            using (var convolver = new PaddedConvolver(convKernel, new Rectangle(0, 0, 0, 0)))
                Tools.WriteToMeltCSV(convolver.Convolve(x), Path.Combine(outputFolder, "psfCircular.csv"));

            //================================================= Reconstruct =============================================================
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);
            var reconstruction = new float[c.GridSize, c.GridSize];
            var fastCD = new FastGreedyCD(totalSize, psf);
            var lambda = 0.50f * fastCD.MaxLipschitz;
            var alpha = 0.2f;

            var residualVis = data.Visibilities;
            for (int cycle = 0; cycle < 5; cycle++)
            {
                Console.WriteLine("in cycle " + cycle);
                var dirtyGrid = IDG.Grid(c, metadata, residualVis, data.UVW, data.Frequencies);
                var dirtyImage = FFT.BackwardFloat(dirtyGrid, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                
                var gradients = Residuals.CalcGradientMap(dirtyImage, corrKernel, totalSize);

                if (cycle == 0)
                {
                    Tools.WriteToMeltCSV(dirtyImage, Path.Combine(outputFolder, "dirty.csv"));
                    Tools.WriteToMeltCSV(gradients, Path.Combine(outputFolder, "gradients.csv"));

                }

                fastCD.Deconvolve(reconstruction, gradients, lambda, alpha, 10000, 1e-5f);

                FFT.Shift(reconstruction);
                var xGrid = FFT.Forward(reconstruction);
                FFT.Shift(reconstruction);
                var modelVis = IDG.DeGrid(c, metadata, xGrid, data.UVW, data.Frequencies);
                residualVis = Visibilities.Substract(data.Visibilities, modelVis, data.Flags);
            }

            //FitsIO.Write(reconstruction, Path.Combine(outputFolder,"xImage.fits"));
            Tools.WriteToMeltCSV(reconstruction, Path.Combine(outputFolder, "elasticNet.csv"));

        }
    }
}
