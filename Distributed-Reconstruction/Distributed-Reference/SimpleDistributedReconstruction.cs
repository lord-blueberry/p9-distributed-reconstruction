using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using MPI;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using Single_Reference;
using static Single_Reference.Common;

using Distributed_Reference.DistributedDeconvolution;


namespace Distributed_Reference
{
    class SimpleDistributedReconstruction
    {
        private class DirtyImage
        {
            public double[,] Image;
            public double MaxSidelobeLevel;

            public DirtyImage(double[,] dirty, double maxLevel)
            {
                Image = dirty;
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

            var imgSection = CalculateLocalImageSection(comm.Rank, comm.Size, c.GridSize, c.GridSize);
            var totalImage = new Common.Rectangle(0, 0, c.GridSize, c.GridSize);

            //calculate psf and prepare for correlation in the Fourier space
            var psf = CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies);
            var psfCut = CutImg(psf);
            Complex[,] PsfCorrelation = null;
            var maxSidelobe = PSF.CalcMaxSidelobe(psf);
            
            if (comm.Rank == 0)
            {
                PsfCorrelation = PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);
            }

            var residualVis = local.Visibilities;
            var xLocal = new double[imgSection.YEnd - imgSection.Y, imgSection.XEnd - imgSection.X];
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                var dirtyImage = ForwardCalculateB(comm, c, metadata, residualVis, local.UVW, local.Frequencies, PsfCorrelation, psfCut, maxSidelobe, watchForward);
                var bLocal = GetImgSection(dirtyImage.Image, imgSection);
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
                    var x = new double[c.GridSize, c.GridSize];
                    StitchImage(totalX, x, comm.Size);
                    FitsIO.Write(x, "xImage_" + cycle + ".fits");
                    FFT.Shift(x);
                    modelGrid = FFT.Forward(x);
                }
                comm.Broadcast(ref modelGrid, 0);

                var modelVis = IDG.DeGrid(c, metadata, modelGrid, local.UVW, local.Frequencies);
                residualVis = IDG.Substract(local.Visibilities, modelVis, local.Flags);
            }

            double[][,] gatherX = null;
            comm.Gather(xLocal, 0, ref gatherX);
            double[,] reconstructed = null;
            if (comm.Rank == 0) 
            {
                reconstructed = new double[c.GridSize, c.GridSize]; ;
                StitchImage(gatherX, reconstructed, comm.Size);
            }
            
            return reconstructed;
        }

        private static Rectangle CalculateLocalImageSection(int nodeId, int nodeCount, int ySize, int xSize)
        {
            var yPatchCount = (int)Math.Floor(Math.Sqrt(nodeCount));
            var xPatchCount = (nodeCount / yPatchCount);

            var yIdx = nodeId / xPatchCount;
            var xIdx = nodeId % xPatchCount;

            var yPatchOffset = yIdx * (ySize / yPatchCount);
            var xPatchOffset = xIdx * (xSize / xPatchCount);

            var yPatchEnd = yIdx + 1 < yPatchCount ? yPatchOffset + ySize / yPatchCount : ySize;
            var xPatchEnd = xIdx + 1 < xPatchCount ? xPatchOffset + xSize / xPatchCount : xSize;

            return new Rectangle(yPatchOffset, xPatchOffset, yPatchEnd, xPatchEnd);
        }

        private static double[,] GetImgSection(double[,] b, Common.Rectangle imgSection)
        {
            var yLen = imgSection.YEnd - imgSection.Y;
            var xLen = imgSection.XEnd - imgSection.X;

            var bLocal = new double[yLen, xLen];
            for (int i = 0; i < yLen; i++)
                for (int j = 0; j < xLen; j++)
                    bLocal[i, j] = b[i + imgSection.Y, j + imgSection.X];

            return bLocal;
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
                var dirtyImage = FFT.Backward(grid_total, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);

                maxSideLobeLevel = maxSidelobe * DebugMethods.GetMax(dirtyImage);
                //remove spheroidal

                image = Common.Residuals.CalculateBMap(dirtyImage, PsfCorrelation, psfCut.GetLength(0), psfCut.GetLength(1));
                watchIdg.Stop();
            }
            comm.Broadcast(ref maxSideLobeLevel, 0);
            comm.Broadcast(ref image, 0);
            return new DirtyImage(image, maxSideLobeLevel);
        }


        private static void StitchImage(double[][,] totalX, double[,] stitched, int nodeCount)
        {
            var yPatchCount = (int)Math.Floor(Math.Sqrt(nodeCount));
            var xPatchCount = (nodeCount / yPatchCount);
            int yOffset = 0;
            for (int yIdx = 0; yIdx < yPatchCount; yIdx++)
            {
                int xOffset = 0;
                int patchIdx = yIdx * yPatchCount;
                for (int xIdx = 0; xIdx < xPatchCount; xIdx++)
                {
                    var patch = totalX[patchIdx+xIdx];
                    for (int y = 0; y < patch.GetLength(0); y++)
                        for (int x = 0; x < patch.GetLength(1); x++)
                            stitched[yOffset + y, xOffset + x] = patch[y, x];
                    xOffset += patch.GetLength(1);
                }
                yOffset += totalX[patchIdx].GetLength(0);
            }
        }


        public static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            double[,] psf = null;
            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf_total = comm.Reduce(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.Backward(psf_total, c.VisibilitiesCount);
                FFT.Shift(psf);
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
