using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using MPI;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using Single_Reference;

using Distributed_Reference.DistributedDeconvolution;


namespace Distributed_Reference
{
    class SimpleDistributedReconstruction
    {
        private class DirtyImage
        {
            public double[,] Dirty;
            public double MaxSidelobeLevel;

            public DirtyImage(double[,] dirty, double maxLevel)
            {
                Dirty = dirty;
                MaxSidelobeLevel = maxLevel;
            }
        }

        public static double[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset local, GriddingConstants c, int maxCycle)
        {
            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackward = new Stopwatch();
            var watchDeconv = new Stopwatch();

            var metadata = Partitioner.CreatePartition(c, local.UVW, local.Frequencies);

            var halfComm = comm.Size / 2;
            var yResOffset = comm.Rank / 2 * (c.GridSize / halfComm);
            var xResOffset = comm.Rank % 2 * (c.GridSize / halfComm);
            var imgSection = new Communication.Rectangle(yResOffset, xResOffset, yResOffset + c.GridSize / halfComm, xResOffset + c.GridSize / halfComm);
            var totalImage = new Communication.Rectangle(0, 0, c.GridSize, c.GridSize);

            //calculate psf and prepare for correlation in the Fourier space
            var psf = CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies);
            var psfCut = CutImg(psf);
            Complex[,] PsfCorrelation = null;
            var maxSidelobe = DebugMethods.GetMaxSidelobeLevel(psf);
            if (comm.Rank == 0)
                PsfCorrelation = GreedyCD2.PadAndInvertPsf(psfCut, c.GridSize, c.GridSize);

            var residualVis = local.Visibilities;
            var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                var dirtyImage = ForwardCalculateB(comm, c, metadata, residualVis, local.UVW, local.Frequencies, PsfCorrelation, psfCut, maxSidelobe, watchForward);
                var bLocal = GetImgSection(dirtyImage.Dirty, imgSection);
                if (comm.Rank == 0)
                    watchDeconv.Start();

                var lambda = 0.8;
                var alpha = 0.05;
                var currentLambda = Math.Max(1.0 / alpha * dirtyImage.MaxSidelobeLevel, lambda);
                var converged = DistributedGreedyCD.DeconvolvePath(comm, imgSection, totalImage, xLocal, bLocal, psfCut, currentLambda, 4.0, alpha, 5, 1000, 2e-5);
                if (comm.Rank == 0)
                {
                    if (converged)
                        Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                    else
                        Console.WriteLine("-------------------------------not converged----------------------");
                }
                comm.Barrier();
                if (comm.Rank == 0)
                    watchDeconv.Stop();

                double[][,] totalX = null;
                comm.Gather(xLocal, 0, ref totalX);
                Complex[,] modelGrid = null;
                if (comm.Rank == 0)
                {
                    watchBackward.Start();
                    var x = StitchX(comm, c, totalX);
                    FitsIO.Write(x, "xImage_" + cycle + ".fits");
                    FFT.Shift(x);
                    modelGrid = FFT.GridFFT(x);
                }
                comm.Broadcast(ref modelGrid, 0);

                var modelVis = IDG.DeGrid(c, metadata, modelGrid, local.UVW, local.Frequencies);
                residualVis = IDG.Substract(local.Visibilities, modelVis, local.Flags);
            }

            double[][,] gatherX = null;
            comm.Gather(xLocal, 0, ref gatherX);
            var reconstructionGlobal = StitchX(comm, c, gatherX);
            
            return reconstructionGlobal;
        }


        private static DirtyImage ForwardCalculateB(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, Complex[,] PsfCorrelation, double[,] psfCut, double maxSidelobe, Stopwatch watchIdg)
        {
            Stopwatch another = new Stopwatch();
            comm.Barrier();
            if (comm.Rank == 0)
            {
                watchIdg.Start();
            }

            var localGrid = IDG.Grid(c, metadata, visibilities, uvw, frequencies);
            double[,] image = null;
            double maxSideLobeLevel = 0.0;
            var grid_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                var dirtyImage = FFT.GridIFFT(grid_total, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);

                maxSideLobeLevel = maxSidelobe * DebugMethods.GetMax(dirtyImage);
                //remove spheroidal

                var dirtyPadded = GreedyCD2.PadResiduals(dirtyImage, psfCut);
                var DirtyPadded = FFT.FFTDebug(dirtyPadded, 1.0);
                var B = IDG.Multiply(DirtyPadded, PsfCorrelation);
                var bPadded = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                image = GreedyCD2.RemovePadding(bPadded, psfCut);
                watchIdg.Stop();
            }
            comm.Broadcast(ref maxSideLobeLevel, 0);
            comm.Broadcast(ref image, 0);
            return new DirtyImage(image, maxSideLobeLevel);
        }


        public static double[,] StitchX(Intracommunicator comm, GriddingConstants c, double[][,] totalX)
        {
            var halfComm = comm.Size / 2;
            var stitched = new double[c.GridSize, c.GridSize];
            int patchIdx = 0;
            for (int patchRows = 0; patchRows < halfComm; patchRows++)
            {
                for (int patchColumns = 0; patchColumns < halfComm; patchColumns++)
                {
                    int yOffset = patchRows * (c.GridSize / halfComm);
                    int xOffset = patchColumns * (c.GridSize / halfComm);
                    var patch = totalX[patchIdx++];
                    for (int y = 0; y < (c.GridSize / halfComm); y++)
                        for (int x = 0; x < (c.GridSize / halfComm); x++)
                            stitched[yOffset + y, xOffset + x] = patch[y, x];
                }
            }

            return stitched;
        }


        private static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            double[,] psf = null;
            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf_total = comm.Reduce(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.GridIFFT(psf_total, c.VisibilitiesCount);
                FFT.Shift(psf);
                Single_Reference.FitsIO.Write(psf, "psf.fits");
                Console.WriteLine("psf Written");
            }
            comm.Broadcast(ref psf, 0);

            return psf;
        }


        private static Complex[,] SequentialSum(Complex[,] a, Complex[,] b)
        {
            for (int y = 0; y < a.GetLength(0); y++)
            {
                for (int x = 0; x < a.GetLength(1); x++)
                {
                    a[y, x] += b[y, x];
                }
            }

            return a;
        }


        private static double[,] CutImg(double[,] image)
        {
            var output = new double[image.GetLength(0) / 2, image.GetLength(1) / 2];
            var yOffset = image.GetLength(0) / 2 - output.GetLength(0) / 2;
            var xOffset = image.GetLength(1) / 2 - output.GetLength(1) / 2;

            for (int y = 0; y < output.GetLength(0); y++)
                for (int x = 0; x < output.GetLength(0); x++)
                    output[y, x] = image[yOffset + y, xOffset + x];
            return output;
        }
    }
}
