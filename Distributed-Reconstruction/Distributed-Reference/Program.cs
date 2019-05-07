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

                var comm = Communicator.world;
                int sum = comm.Rank;
                var total = comm.Reduce(sum, (a, b) => a + b, 0);
                if(comm.Rank == 0)
                {
                    Console.WriteLine("testing mpi");
                    Console.WriteLine(total);
                }
                //READ DATA
                /*
                var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
                var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
                var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

                var visCount2 = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCount2++;
                */
                var frequencies = FitsIO.ReadFrequencies(@"freq.fits");
                var uvw = FitsIO.ReadUVW(@"uvw.fits");
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
                var visibilities = FitsIO.ReadVisibilities(@"vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);
                var visibilitiesCount = visibilities.Length;

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
                            flagstmp[i, j, k] = flags[blOffset + i, j, k];
                        }
                        uvwtmp[i, j, 0] = uvw[blOffset + i, j, 0];
                        uvwtmp[i, j, 1] = uvw[blOffset + i, j, 1];
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
                flags = flagstmp;

                int gridSize = 1024;
                int subgridsize = 16;
                int kernelSize = 8;
                //cell = image / grid
                int max_nr_timesteps = 512;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;

                comm.Barrier();
                var watchTotal = new Stopwatch();
                var watchForward = new Stopwatch();
                var watchBackward = new Stopwatch();
                var watchDeconv = new Stopwatch();
                if (comm.Rank == 0)
                {
                    Console.WriteLine("Done Reading, Start Gridding");
                    watchTotal.Start();
                }
                var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psf = CalculatePSF(comm, c, metadata, uvw, flags, frequencies);
                if (comm.Rank == 0)
                {
                    Console.WriteLine("Done  PSF");
                }

                var halfComm = comm.Size / 2;
                var yResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank / 2 * (gridSize / halfComm);

                var residualVis = visibilities;
                var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
                for (int cycle=0; cycle < 7; cycle++)
                {
                    var imageLocal = Forward(comm, c, metadata, residualVis, uvw, frequencies, watchForward);
                    if (comm.Rank == 0)
                    {
                        Console.WriteLine("Done  forward part of Cycle "+cycle);
                        watchDeconv.Start();
                        //FitsIO.Write(imageLocal, "residual"+cycle+".fits");
                    }
                    CDClean.Deconvolve(xLocal, imageLocal, psf, 0.1 / (10 * (cycle + 1)), 5, yResOffset, xResOffset);
                    comm.Barrier();
                    if (comm.Rank == 0)
                    {
                        watchDeconv.Stop();
                    }

                    double[][,] totalX = null;
                    comm.Gather<double[,]>(xLocal, 0, ref totalX);
                    Complex[,] modelGrid = null;
                    if (comm.Rank == 0)
                    {
                        watchBackward.Start();
                        var x = StitchX(comm, c, totalX);
                        //FitsIO.Write(x, "xImage_"+cycle+".fits");

                        FFT.Shift(x);
                        modelGrid = FFT.GridFFT(x);
                    }
                    comm.Broadcast(ref modelGrid, 0);

                    var modelVis = IDG.DeGrid(c, metadata, modelGrid, uvw, frequencies);
                    residualVis = IDG.Substract(visibilities, modelVis, flags);
                    if (comm.Rank == 0)
                        watchBackward.Stop();

                    var modelImg = Forward(comm, c, metadata, modelVis, uvw, frequencies, watchForward);
                    if(comm.Rank == 0)
                    {
                        //FitsIO.Write(modelImg, "model_" + cycle + ".fits");
                    }
                }

                if (comm.Rank == 0)
                {
                    watchTotal.Stop();
                    
                    FitsIO.Write(psf, "psf.fits");
                    var timetable = "total elapsed: " + watchTotal.Elapsed;
                    timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
                    timetable += "\n" + "idg backwards elapsed: " + watchBackward.Elapsed;
                    timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
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

        public static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            double[,] psf = null;

            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.GridIFFT(psf_total, c.VisibilitiesCount);
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
                image = FFT.GridIFFT(grid_total, c.VisibilitiesCount);
                FFT.Shift(image);
                watchIdg.Stop();

                //Single_Reference.FitsIO.Write(image, "dirty.fits");
                //Console.WriteLine("fits Written");

                //remove spheroidal
            }

            comm.Broadcast<double[,]>(ref image, 0);
            return image;
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
