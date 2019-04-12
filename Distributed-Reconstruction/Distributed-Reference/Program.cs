using System;
using MPI;
using System.Diagnostics;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static System.Math;
using System.Numerics;
using System.Collections.Generic;

namespace Distributed_Reference
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);
                //System.Threading.Thread.Sleep(17000);
                
                Console.WriteLine("Hello World! from rank " + Communicator.world.Rank + " (running on " + MPI.Environment.ProcessorName + ")");
                var comm = Communicator.world;

                //READ DATA
                var frequencies = Single_Reference.FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\freq.fits");
                var uvw = Single_Reference.FitsIO.ReadUVW(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\uvw.fits");
                var visibilities = Single_Reference.FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-distributed-reconstruction\Distributed-Reconstruction\p9-data\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);

                var nrBaselines = uvw.GetLength(0) / comm.Size;
                var nrFrequencies = frequencies.Length;
                var uvwtmp = new double[nrBaselines, uvw.GetLength(1), 3];
                var vistmp = new Complex[nrBaselines, uvw.GetLength(1), nrFrequencies];
                var freqtmp = new double[nrFrequencies];
                var blOffset = uvw.GetLength(0) / comm.Size * comm.Rank;
                for (int i = 0; i < nrBaselines; i++)
                {
                    for (int j = 0; j < uvw.GetLength(1); j++)
                    {
                        for (int k = 0; k < nrFrequencies; k++)
                        {
                            vistmp[i, j, k] = visibilities[blOffset + i, j, k];
                        }
                        uvwtmp[i, j, 0] = uvw[blOffset + i, j, 0];
                        uvwtmp[i, j, 1] = uvw[blOffset+ i, j, 1];
                        uvwtmp[i, j, 2] = uvw[blOffset + i, j, 2];
                    }
                }

                for (int i = 0; i < nrFrequencies; i++)
                {
                    freqtmp[i] = frequencies[i];
                }

                var watch = new Stopwatch();
                var watchIdg = new Stopwatch();
                if (comm.Rank == 0)
                    watch.Start();


                uvw = uvwtmp;
                visibilities = vistmp;
                frequencies = freqtmp;

                int gridSize = 256;
                int subgridsize = 64;
                int kernelSize = 32;
                //cell = image / grid
                int max_nr_timesteps = 256;
                double cellSize = 0.5 / 3600.0 * PI / 180.0;
                if (comm.Rank == 0)
                    watchIdg.Start();
                var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

                var psf = CalculatePSF(comm, c, metadata, uvw, frequencies, visibilities.LongLength * comm.Size);

                
                var imagePatch = Forward(comm, c, metadata, visibilities, uvw, frequencies);
                if (comm.Rank == 0)
                {
                    watchIdg.Stop();
                    Console.WriteLine(watchIdg.Elapsed);
                }
                    

                var localX = new double[imagePatch.GetLength(0), imagePatch.GetLength(1)];

                if (comm.Rank == 0)
                {
                    Console.WriteLine("deconvolve");
                }
                CDClean.Deconvolve(localX, imagePatch, psf, 1.9, 2);

                comm.Barrier();
                double[][,] totalX = null;
                comm.Gather<double[,]>(localX, 0, ref totalX);
                if (comm.Rank == 0)
                {
                    var reconstructed = new double[gridSize, gridSize];
                    int patchIdx = 0;
                    var oneSide = comm.Size / 2;
                    for (int patchRows = 0; patchRows < oneSide; patchRows++)
                    {
                        for (int patchColumns = 0; patchColumns < oneSide; patchColumns++)
                        {
                            int yOffset = patchRows * (gridSize / oneSide);
                            int xOffset = patchColumns * (gridSize / oneSide);
                            var patch = totalX[patchIdx++];
                            for (int y = 0; y < (gridSize / oneSide); y++)
                                for (int x = 0; x < (gridSize / oneSide); x++)
                                    reconstructed[yOffset + y, xOffset + x] = patch[y, x];
                        }
                    }

                    watch.Stop();
                    Console.WriteLine("");
                    Console.WriteLine("");
                    Console.WriteLine(watchIdg.Elapsed);
                    Console.WriteLine(watch.Elapsed);
                    Console.WriteLine("");
                    Console.WriteLine("");
                    Single_Reference.FitsIO.Write(reconstructed, "xImge.fits");
                }
                

            }
        }

        public static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            double[,] psf = null;

            var localGrid = IDG.GridPSF(c, metadata, uvw, frequencies, visibilitiesCount);
            var psf_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.GridIFFT(psf_total);
                FFT.Shift(psf);
                psf = CutImg(psf);
                //Single_Reference.FitsIO.Write(psf, "psf.fits");
                //Console.WriteLine("psf Written");
                
            }
            comm.Broadcast(ref psf, 0);

            return psf;
        }

        public static double[,] Forward(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            double[][,] patches = null;

            var watchIDG = new Stopwatch();
            if (comm.Rank == 0)
                watchIDG.Start();
            var localGrid = IDG.Grid(c, metadata, visibilities, uvw, frequencies);
            if (comm.Rank == 0)
            {
                watchIDG.Stop();
                Console.WriteLine("IDG Pure");
                Console.WriteLine(watchIDG.Elapsed);
            }
                
            var grid_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                var image = FFT.GridIFFT(grid_total);
                FFT.Shift(image);
                
                //Single_Reference.FitsIO.Write(image, "dirty.fits");
                //Console.WriteLine("fits Written");

                //remove spheroidal

                //create patches
                patches = new double[comm.Size][,];
                int patchIdx = 0;
                var oneSide = comm.Size / 2;
                for (int patchRows = 0; patchRows < oneSide; patchRows++)
                {
                    for(int patchColumns = 0; patchColumns < oneSide; patchColumns++)
                    {
                        int yOffset = patchRows * (image.GetLength(0) / oneSide);
                        int xOffset = patchColumns * (image.GetLength(1) / oneSide);
                        var patch = new double[(image.GetLength(0) / oneSide), (image.GetLength(1) / oneSide)];
                        patches[patchIdx++] = patch;
                        for(int y = 0; y < (image.GetLength(0) / oneSide); y++)
                            for(int x = 0; x < (image.GetLength(1) / oneSide); x++)
                                patch[y, x] = image[yOffset + y, xOffset + x];
                    }
                }
            }

            var localPatch = comm.Scatter<double[,]>(patches, 0);
            return localPatch;
        }

        private static double[,] CutImg(double[,] image)
        {
            var output = new double[128, 128];
            var yOffset = image.GetLength(0) / 2 - output.GetLength(0) / 2;
            var xOffset = image.GetLength(1) / 2 - output.GetLength(1) / 2;

            for (int y = 0; y < output.GetLength(0); y++)
                for (int x = 0; x < output.GetLength(0); x++)
                    output[y, x] = image[yOffset + y, xOffset + x];
            return output;
        }

        public static Complex[,] SequentialSum(Complex[,] a, Complex[,] b)
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
