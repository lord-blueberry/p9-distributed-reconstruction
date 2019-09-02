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
using Distributed_Reference.DistributedDeconvolution;
using static Single_Reference.Common;

namespace Distributed_Reference
{
    class RunningMethods
    { 

        public static void RunTest(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);
                System.Threading.Thread.Sleep(17000);

                var comm = Communicator.world;
                int sum = comm.Rank;
                var total = comm.Reduce(sum, (a, b) => a + b, 0);
                if (comm.Rank == 0)
                {
                    Console.WriteLine("testing mpi");
                    Console.WriteLine(total);
                }
            }
        }

        public static void RunTinyMeerKAT(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);

                var comm = Communicator.world;
                var beginIdx = comm.Rank * 8 / comm.Size;

                Console.WriteLine("Rank {0} reads idx {1} ", comm.Rank, beginIdx);
                var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
                var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw" + beginIdx + ".fits");
                var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags" + beginIdx + ".fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis" + beginIdx + ".fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

                for (int i = beginIdx + 1; i < beginIdx + 8 / comm.Size; i++)
                {
                    Console.WriteLine("Rank {0} reads idx {1} ", comm.Rank, i);
                    var uvw0 = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw" + i + ".fits");
                    var flags0 = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                    var visibilities0 = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis" + i + ".fits", uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, 2.0);
                    uvw = FitsIO.Stitch(uvw, uvw0);
                    flags = FitsIO.Stitch(flags, flags0);
                    visibilities = FitsIO.Stitch(visibilities, visibilities0);
                }

                var visCountLocal = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCountLocal++;

                var visibilitiesCount = comm.Allreduce(visCountLocal, (x, y) => x+y);

                int gridSize = 1024;
                int subgridsize = 16;
                int kernelSize = 4;
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
                var halfComm = comm.Size / 2;
                var yResOffset = comm.Rank / 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var imgSection = new Common.Rectangle(yResOffset, xResOffset, yResOffset + gridSize / halfComm, xResOffset + gridSize / halfComm);
                var totalImage = new Common.Rectangle(0, 0, c.GridSize, c.GridSize);

                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psf = CalculatePSF(comm, c, metadata, uvw, flags, frequencies);
                var psfCut = CutImg(psf);
                var maxSidelobe = Common.PSF.CalcMaxSidelobe(psf);
                psf = null;
                Complex[,] PsfCorrelation = null;
                if (comm.Rank == 0)
                    PsfCorrelation = Common.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);

                var residualVis = visibilities;
                var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
                var maxCycle = 5;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    var forwardPass = ForwardCalculateB(comm, c, metadata, residualVis, uvw, frequencies, PsfCorrelation, psfCut, maxSidelobe, watchForward);
                    var bLocal = GetImgSection(forwardPass.Item1, imgSection);
                    if (comm.Rank == 0)
                        watchDeconv.Start();

                    var lambda = 0.8;
                    var alpha = 0.05;
                    var currentLambda = Math.Max(1.0 / alpha * forwardPass.Item2, lambda);
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
                    comm.Gather<double[,]>(xLocal, 0, ref totalX);
                    Complex[,] modelGrid = null;
                    if (comm.Rank == 0)
                    {
                        watchBackward.Start();
                        var x = StitchX(comm, c, totalX);
                        FitsIO.Write(x, "xImage_" + cycle + ".fits");
                        FFT.Shift(x);
                        modelGrid = FFT.Forward(x);
                    }
                    comm.Broadcast(ref modelGrid, 0);

                    var modelVis = IDG.DeGrid(c, metadata, modelGrid, uvw, frequencies);
                    residualVis = IDG.Substract(visibilities, modelVis, flags);
                    if (comm.Rank == 0)
                        watchBackward.Stop();
                }

                if (comm.Rank == 0)
                {
                    watchTotal.Stop();

                    var timetable = "total elapsed: " + watchTotal.Elapsed;
                    timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
                    timetable += "\n" + "idg backwards elapsed: " + watchBackward.Elapsed;
                    timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
                    File.WriteAllText("watches_mpi.txt", timetable);
                }
            }
        }

        public static void RunSubsetTinyMeerKAT(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);

                var comm = Communicator.world;
                var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";
                /*
                var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\freq.fits");
                var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\uvw0.fits");
                var flags = FitsIO.ReadFlags(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0); //norm by 2.0 because we combine polarization XX and YY to I
                */

                var frequencies = FitsIO.ReadFrequencies(@"freq.fits");
                var uvw = FitsIO.ReadUVW(@"uvw0.fits");
                var flags = FitsIO.ReadFlags(@"flags0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                var visibilities = FitsIO.ReadVisibilities(@"vis0.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

                var visCount2 = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCount2++;
                var visibilitiesCount = visCount2;

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
                int kernelSize = 4;
                int max_nr_timesteps = 512;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

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

                var lambda = 0.5f;
                var alpha = 0.4f;



                var halfComm = comm.Size / 2;
                var yResOffset = comm.Rank / 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var imgSection = new Common.Rectangle(yResOffset, xResOffset, yResOffset + gridSize / halfComm, xResOffset + gridSize / halfComm);
                var totalImage = new Common.Rectangle(0, 0, c.GridSize, c.GridSize);

                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psf = CalculatePSF(comm, c, metadata, uvw, flags, frequencies);
                var psfCut = CutImg(psf);
                Complex[,] PsfCorrelation = null;
                if (comm.Rank == 0)
                    PsfCorrelation = Common.PSF.CalculateFourierCorrelation(psfCut, c.GridSize, c.GridSize);

                var integral = Common.PSF.CalcPSFScan(psfCut);
                var aMapLocal = new double[imgSection.YEnd, imgSection.XEnd];
                for (int y = imgSection.Y; y < imgSection.YEnd; y++)
                    for (int x = imgSection.X; x < imgSection.XEnd; x++)
                    {
                        var yPixel = y - imgSection.Y;
                        var xPixel = x - imgSection.X;
                        aMapLocal[yPixel, xPixel] = Common.PSF.QueryScan(integral, y, x, totalImage.YEnd, totalImage.XEnd);
                    }

                var residualVis = visibilities;
                var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
                var maxCycle = 5;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    var forwardPass = ForwardCalculateB(comm, c, metadata, residualVis, uvw, frequencies, PsfCorrelation, psfCut, 0.0, watchForward);
                    var bLocal = GetImgSection(forwardPass.Item1, imgSection);
                    var bCopy = new double[c.GridSize, c.GridSize];
                    if (comm.Rank == 0)
                        watchDeconv.Start();

                    var lambdaStart = 2.5;
                    var lambdaEnd = 0.1;
                    var lambda = lambdaStart - (lambdaStart - lambdaEnd) / (maxCycle) * (cycle + 1);
                    var converged = DistributedGreedyCD.Deconvolve(comm, imgSection, totalImage, xLocal, aMapLocal, bLocal, psfCut, lambda, 0.4, 10000);
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
                    comm.Gather<double[,]>(xLocal, 0, ref totalX);
                    Complex[,] modelGrid = null;
                    if (comm.Rank == 0)
                    {
                        watchBackward.Start();
                        var x = StitchX(comm, c, totalX);
                        FitsIO.Write(x, "xImage_" + cycle + ".fits");
                        FFT.Shift(x);
                        modelGrid = FFT.Forward(x);
                    }
                    comm.Broadcast(ref modelGrid, 0);

                    var modelVis = IDG.DeGrid(c, metadata, modelGrid, uvw, frequencies);
                    residualVis = IDG.Substract(visibilities, modelVis, flags);
                    if (comm.Rank == 0)
                        watchBackward.Stop();
                }

                if (comm.Rank == 0)
                {
                    watchTotal.Stop();

                    var timetable = "total elapsed: " + watchTotal.Elapsed;
                    timetable += "\n" + "idg forward elapsed: " + watchForward.Elapsed;
                    timetable += "\n" + "idg backwards elapsed: " + watchBackward.Elapsed;
                    timetable += "\n" + "devonvolution: " + watchDeconv.Elapsed;
                    File.WriteAllText("watches_mpi.txt", timetable);
                }
            }
        }

        public static void RunSimulated(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);
                //System.Threading.Thread.Sleep(17000);

                var comm = Communicator.world;
                //READ DATA

                var folder = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
                var fullData = DistributedData.LoadSimulated(folder);
                var data = DistributedData.SplitDataAmongNodes(comm, fullData);

                int gridSize = 128;
                int subgridsize = 16;
                int kernelSize = 8;
                int max_nr_timesteps = 512;
                double cellSize = 2.0 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

                comm.Barrier();
                if (comm.Rank == 0)
                {
                    Console.WriteLine("Done Reading, Starting reconstruction");
                }

                var reconstruction = SimpleDistributedReconstruction.Reconstruct(comm, data, c, 5, 0.1f, 0.8f);

                if (comm.Rank == 0)
                {
                    FitsIO.Write(reconstruction, "simulatedReconstruction.fits");
                }
            }
        }
    }
}
