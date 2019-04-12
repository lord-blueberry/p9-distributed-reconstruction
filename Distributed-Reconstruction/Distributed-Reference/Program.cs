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

                uvw = uvwtmp;
                visibilities = vistmp;
                frequencies = freqtmp;

                var watchTotal = new Stopwatch();
                var watchNufft = new Stopwatch();
                var watchIdg = new Stopwatch();
                if (comm.Rank == 0)
                    watchTotal.Start();

                int gridSize = 256;
                int subgridsize = 32;
                int kernelSize = 8;
                //cell = image / grid
                int max_nr_timesteps = 256;
                double cellSize = 0.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psf = CalculatePSF(comm, c, metadata, uvw, frequencies, visibilities.LongLength * comm.Size);
                var imageLocal = Forward(comm, c, metadata, visibilities, uvw, frequencies, watchIdg);

                var halfComm = comm.Size / 2;
                var localX = new double[imageLocal.GetLength(0) / halfComm, imageLocal.GetLength(1)/ halfComm];

                if (comm.Rank == 0)
                    Console.WriteLine("deconvolve");

                var yResOffset = comm.Rank / 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank % 2 * (gridSize / halfComm);
                CDClean.Deconvolve(localX, imageLocal, psf, 2.0, 5, yResOffset, xResOffset);

                comm.Barrier();
                double[][,] totalX = null;
                comm.Gather<double[,]>(localX, 0, ref totalX);
                if (comm.Rank == 0)
                {
                    var reconstructed = new double[gridSize, gridSize];
                    int patchIdx = 0;
                    for (int patchRows = 0; patchRows < halfComm; patchRows++)
                    {
                        for (int patchColumns = 0; patchColumns < halfComm; patchColumns++)
                        {
                            int yOffset = patchRows * (gridSize / halfComm);
                            int xOffset = patchColumns * (gridSize / halfComm);
                            var patch = totalX[patchIdx++];
                            for (int y = 0; y < (gridSize / halfComm); y++)
                                for (int x = 0; x < (gridSize / halfComm); x++)
                                    reconstructed[yOffset + y, xOffset + x] = patch[y, x];
                        }
                    }

                    watchTotal.Stop();
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

        public static double[,] Forward(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, Stopwatch watchIdg)
        {
            if (comm.Rank == 0)
                watchIdg.Start();

            var localGrid = IDG.Grid(c, metadata, visibilities, uvw, frequencies);
            double[,] image = null;
            var grid_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                image = FFT.GridIFFT(grid_total);
                FFT.Shift(image);
                watchIdg.Stop();
                
                //Single_Reference.FitsIO.Write(image, "dirty.fits");
                //Console.WriteLine("fits Written");

                //remove spheroidal

            }

            comm.Broadcast<double[,]>(ref image, 0);
            return image;
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
