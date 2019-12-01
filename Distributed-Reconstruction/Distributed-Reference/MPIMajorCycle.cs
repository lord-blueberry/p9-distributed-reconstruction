using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using MPI;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using Core;
using static Core.Common;
using System.IO;


namespace DistributedReconstruction
{
    class MPIMajorCycle
    {
        private class DirtyImage
        {
            public float[,] Image;
            public float MaxSidelobeLevel;

            public DirtyImage(float[,] dirty, float maxLevel)
            {
                Image = dirty;
                MaxSidelobeLevel = maxLevel;
            }
        }

        public static float[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset local, GriddingConstants c, int maxCycle, float lambda, float alpha, int iterPerCycle = 1000,bool usePathDeconvolution = false)
        {
            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackward = new Stopwatch();
            var watchDeconv = new Stopwatch();
            watchTotal.Start();

            var metadata = Partitioner.CreatePartition(c, local.UVW, local.Frequencies);

            var patchSize = CalculateLocalImageSection(comm.Rank, comm.Size, c.GridSize, c.GridSize);
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);

            //calculate psf and prepare for correlation in the Fourier space
            var psf = CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies);
            Complex[,] PsfCorrelation = null;
            var maxSidelobe = PSF.CalcMaxSidelobe(psf);
            lambda = (float)(lambda * PSF.CalcMaxLipschitz(psf));
                
            StreamWriter writer = null;
            if (comm.Rank == 0)
            {
                FitsIO.Write(psf, "psf.fits");
                Console.WriteLine("done PSF gridding ");
                PsfCorrelation = PSF.CalcPaddedFourierCorrelation(psf, totalSize);
                writer = new StreamWriter(comm.Size + "runtimestats.txt");
            }

            var deconvovler = new MPIGreedyCD(comm, totalSize, patchSize, psf);

            var residualVis = local.Visibilities;
            var xLocal = new float[patchSize.YEnd - patchSize.Y, patchSize.XEnd - patchSize.X];
            
            
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                if (comm.Rank == 0)
                    Console.WriteLine("cycle " + cycle);
                var dirtyImage = ForwardCalculateB(comm, c, metadata, residualVis, local.UVW, local.Frequencies, PsfCorrelation, psf, maxSidelobe, watchForward);
                
                var bLocal = GetImgSection(dirtyImage.Image, patchSize);

                MPIGreedyCD.Statistics lastRun;
                if (usePathDeconvolution)
                {
                    var currentLambda = Math.Max(1.0f / alpha * dirtyImage.MaxSidelobeLevel, lambda);
                    lastRun = deconvovler.DeconvolvePath(xLocal, bLocal, currentLambda, 4.0f, alpha, 5, iterPerCycle, 2e-5f);
                } else
                {
                    lastRun = deconvovler.Deconvolve(xLocal, bLocal, lambda, alpha, iterPerCycle, 1e-5f);
                }

                if (comm.Rank == 0)
                {
                    WriteToFile(cycle, lastRun, writer);
                    if (lastRun.Converged)
                        Console.WriteLine("-----------------------------CONVERGED!!!!------------------------");
                    else
                        Console.WriteLine("-------------------------------not converged----------------------");
                }
                comm.Barrier();
                if (comm.Rank == 0)
                    watchDeconv.Stop();

                float[][,] totalX = null;
                comm.Gather(xLocal, 0, ref totalX);
                Complex[,] modelGrid = null;
                if (comm.Rank == 0)
                {
                    watchBackward.Start();
                    var x = new float[c.GridSize, c.GridSize];
                    StitchImage(totalX, x, comm.Size);
                    FitsIO.Write(x, "xImage_" + cycle + ".fits");
                    FFT.Shift(x);
                    modelGrid = FFT.Forward(x);
                }
                comm.Broadcast(ref modelGrid, 0);

                var modelVis = IDG.DeGrid(c, metadata, modelGrid, local.UVW, local.Frequencies);
                residualVis = Visibilities.Substract(local.Visibilities, modelVis, local.Flags);
            }
            writer.Close();

            float[][,] gatherX = null;
            comm.Gather(xLocal, 0, ref gatherX);
            float[,] reconstructed = null;
            if (comm.Rank == 0) 
            {
                reconstructed = new float[c.GridSize, c.GridSize]; ;
                StitchImage(gatherX, reconstructed, comm.Size);
            }
            
            return reconstructed;
        }

        private static void WriteToFile(int cycle, MPIGreedyCD.Statistics run, StreamWriter writer)
        {
            writer.WriteLine(cycle + ";" + run.IterationsRun + ";" + run.ElapsedMilliseconds.TotalSeconds);
            writer.Flush();
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

        private static float[,] GetImgSection(float[,] b, Rectangle imgSection)
        {
            var yLen = imgSection.YEnd - imgSection.Y;
            var xLen = imgSection.XEnd - imgSection.X;

            var bLocal = new float[yLen, xLen];
            for (int i = 0; i < yLen; i++)
                for (int j = 0; j < xLen; j++)
                    bLocal[i, j] = b[i + imgSection.Y, j + imgSection.X];

            return bLocal;
        }

        private static DirtyImage ForwardCalculateB(Intracommunicator comm, GriddingConstants c, List<List<Subgrid>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, Complex[,] PsfCorrelation, float[,] psfCut, float maxSidelobe, Stopwatch watchIdg)
        {
            Stopwatch another = new Stopwatch();
            comm.Barrier();
            if (comm.Rank == 0)
            {
                watchIdg.Start();
            }

            var localGrid = IDG.Grid(c, metadata, visibilities, uvw, frequencies);
            float[,] image = null;
            float maxSideLobeLevel = 0.0f;
            var grid_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                var dirtyImage = FFT.BackwardFloat(grid_total, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                if (comm.Rank == 0)
                    FitsIO.Write(dirtyImage, "dirtyImage.fits");
                maxSideLobeLevel = maxSidelobe * Residuals.GetMax(dirtyImage);
                //remove spheroidal

                image = Residuals.CalcGradientMap(dirtyImage, PsfCorrelation, new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
                watchIdg.Stop();
            }
            comm.Broadcast(ref maxSideLobeLevel, 0);
            comm.Broadcast(ref image, 0);
            return new DirtyImage(image, maxSideLobeLevel);
        }


        private static void StitchImage(float[][,] totalX, float[,] stitched, int nodeCount)
        {
            for(int i = 0; i < totalX.Length; i++)
            {
                var targetRectangle = CalculateLocalImageSection(i, nodeCount, stitched.GetLength(0), stitched.GetLength(1));
                for (int yIdx = 0; yIdx < targetRectangle.YExtent();yIdx++)
                    for(int xIdx = 0; xIdx < targetRectangle.XExtent();xIdx++)
                    {
                        stitched[yIdx + targetRectangle.Y, xIdx + targetRectangle.X] = totalX[i][yIdx, xIdx];
                    }
            }
        }

        private static float[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<Subgrid>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            float[,] psf = null;
            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf_total = comm.Reduce(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.BackwardFloat(psf_total, c.VisibilitiesCount);
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
    }
}
