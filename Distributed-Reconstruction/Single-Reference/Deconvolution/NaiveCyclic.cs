using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;
using static System.Math;

namespace Single_Reference.Deconvolution
{
    class NaiveCyclic
    {
        public static void Run()
        {
            var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
            var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset

            var visibilitiesCount = flags.Length;
            int gridSize = 64;
            int subgridsize = 16;
            int kernelSize = 8;
            int max_nr_timesteps = 64;
            double cellSize = 2.0 / 3600.0 * PI / 180.0;
            var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf = FFT.GridIFFT(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);
            var maxPsf = psf[gridSize / 2, gridSize / 2];
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    psf[i, j] = psf[i, j] / maxPsf;
            FitsIO.Write(psf, "psf.fits");

            var truth = new double[64, 64];
            truth[40, 50] = 1.5;
            truth[0, 0] = 1.7;
            var dirty = NaiveGreedyCD.ConvolveFFTPadded(truth, psf);
            FitsIO.Write(truth, "truth.fits");
            FitsIO.Write(dirty, "dirty.fits");

        }

        public static void Deconv(double[,] xImage, double[,] res, double[,] psf, double lambda, int maxIter= 10, int maxInnter = 10)
        {

        }
    }
}
