using System;
using MPI;
using System.Diagnostics;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using Single_Reference;
using static System.Math;
using System.Numerics;
using System.Collections.Generic;

using System.IO;

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
                var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
                var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
                //var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
                double norm = 2.0 * uvw.GetLength(0) * uvw.GetLength(1) * frequencies.Length;
                var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);
                /*
                var frequencies = Single_Reference.FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
                var uvw = Single_Reference.FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
                double norm = 2.0 * uvw.GetLength(0) * uvw.GetLength(1) * frequencies.Length;
                var visibilities = Single_Reference.FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);
                */

                var nrBaselines = uvw.GetLength(0) / comm.Size;
                var nrFrequencies = frequencies.Length;
                var uvwtmp = new double[nrBaselines, uvw.GetLength(1), 3];
                var vistmp = new Complex[nrBaselines, uvw.GetLength(1), nrFrequencies];
                var freqtmp = new double[nrFrequencies];
                var flagstmp = new bool[nrBaselines, uvw.GetLength(1), nrFrequencies];
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
                        uvwtmp[i, j, 1] = uvw[blOffset + i, j, 1];
                        uvwtmp[i, j, 2] = uvw[blOffset + i, j, 2];

                        //flags
                    }
                }

                for (int i = 0; i < nrFrequencies; i++)
                {
                    freqtmp[i] = frequencies[i];
                }

                uvw = uvwtmp;
                visibilities = vistmp;
                frequencies = freqtmp;
                flags = flagstmp;

                int gridSize = 1024;
                int subgridsize = 16;
                int kernelSize = 8;
                //cell = image / grid
                int max_nr_timesteps = 256;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;

                comm.Barrier();
                var watchTotal = new Stopwatch();
                var watchNufft = new Stopwatch();
                var watchIdg = new Stopwatch();
                if (comm.Rank == 0)
                {
                    Console.WriteLine("Done Reading, Start Gridding");
                    watchTotal.Start();
                    watchNufft.Start();
                }
                var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psf = CalculatePSF(comm, c, metadata, uvw, flags, frequencies, visibilities.LongLength * comm.Size);
                var imageLocal = Forward(comm, c, metadata, visibilities, uvw, frequencies, watchIdg);

                if (comm.Rank == 0)
                {
                    //Single_Reference.FitsIO.Write(imageLocal, "0_dirty.fits");
                    watchNufft.Stop();
                    Console.WriteLine("deconvolve");
                }
                    
                var halfComm = comm.Size / 2;
                var localX = new double[imageLocal.GetLength(0) / halfComm, imageLocal.GetLength(1) / halfComm];

                var yResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank / 2 * (gridSize / halfComm);
                CDClean.Deconvolve(localX, imageLocal, psf, 1.0, 2, yResOffset, xResOffset);

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
                    FitsIO.Write(imageLocal, "residual_0.fits");
                    FitsIO.Write(reconstructed, "xImage.fits");
                    var timetable = "total elapsed: " + watchTotal.Elapsed;
                    timetable += "\n" + "nufft elapsed: " + watchNufft.Elapsed;
                    timetable += "\n" + "idg elapsed: " + watchIdg.Elapsed;
                    File.WriteAllText("watches_mpi.txt", timetable);
                }
                /*
                ExchangeNonZero(comm, localX, imageLocal, psf, yResOffset, xResOffset);
                CDClean.Deconvolve(localX, imageLocal, psf, 2.0, 5, yResOffset, xResOffset);
                comm.Barrier();
                totalX = null;
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
                    Single_Reference.FitsIO.Write(reconstructed, "xImage2.fits");
                    var timetable = "total elapsed: " + watchTotal.Elapsed;
                    timetable += "\n" + "nufft elapsed: " + watchNufft.Elapsed;
                    timetable += "\n" + "idg elapsed: " + watchIdg.Elapsed;
                    File.WriteAllText("watches.txt", timetable);
                }*/

            }
        }

        public static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies, long visibilitiesCount)
        {
            double[,] psf = null;

            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies, visibilitiesCount);
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

        private static void ExchangeNonZero(Intracommunicator comm, double[,] xImage, double[,] residual, double[,] psf,  int yResOffset, int xResOffset)
        {
            var xNonZero = new List<Tuple<int, int, double>>();
            for(int y = 0; y < xImage.GetLength(0); y++)
            {
                for(int x = 0; x < xImage.GetLength(1); x++)
                {
                    if(xImage[y, x] > 0.0)
                    {
                        xNonZero.Add(new Tuple<int, int, double>(y + yResOffset, x + xResOffset, xImage[y, x]));
                    }
                }
            }

            var globalNonZero = comm.Allgather(xNonZero);
            for(int i = 0; i < comm.Size; i++)
            {
                if(i != comm.Rank)
                {
                    var otherNonZero = globalNonZero[i];
                    foreach(var t in otherNonZero)
                    {
                        CDClean.ModifyResidual(residual, psf, t.Item1, t.Item2, t.Item3);
                    }
                }
            }
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
