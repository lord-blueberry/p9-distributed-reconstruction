using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.Deconvolution;
using Single_Reference.IDGSequential;

namespace Single_Reference
{
    class DeconvTest
    {
        public static void TestMore()
        {
            DebugMethods.PrintFits();

            var truth = new double[64, 64];
            truth[30, 30] = 1.0;
            truth[30, 31] = 1.0;

            var psf = FitsIO.ReadBeam("psf.fits");
            var psfGrid = FFT.ForwardFFTDebug(psf, 1.0);

            var tGrid = FFT.ForwardFFTDebug(truth, 1.0);
            var dGrid = IDG.Multiply(tGrid, psfGrid);
            var dirty = FFT.ForwardIFFTDebug(dGrid, 64 * 64);
            FFT.Shift(dirty);
            FitsIO.Write(dirty, "dirty_truth.fits");

            var bGrid = IDG.Multiply(dGrid, psfGrid);
            var b = FFT.ForwardIFFTDebug(bGrid, 64 * 64);
            //FFT.Shift(b);
            FitsIO.Write(b, "b_truth.fits");

            var psf2 = IDG.Multiply(psfGrid, psfGrid);
            var pfsImg2 = FFT.ForwardIFFTDebug(psf2, 64 * 64);
            FFT.Shift(pfsImg2);
            FitsIO.Write(pfsImg2, "psf_img2.fits");

            var xImage = new double[dirty.GetLength(0), dirty.GetLength(0)];
            var converged = GreedyCD.Deconvolve(xImage, b, pfsImg2, 0.4, 0.0, 200000);
            FitsIO.Write(xImage, "xImage.fits");

            var hello = FFT.ForwardFFTDebug(xImage, 1.0);
            hello = IDG.Multiply(hello, psfGrid);
            var hImg = FFT.ForwardIFFTDebug(hello, 64 * 64);
            FFT.Shift(hImg);
            FitsIO.Write(hImg, "modelDirty.fits");
            //FitsIO.Write(b, "b_output.fits");
        }
    }
}
