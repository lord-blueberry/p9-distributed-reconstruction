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
                System.Threading.Thread.Sleep(17000);

                var comm = Communicator.world;
                int sum = comm.Rank;
                var total = comm.Reduce(sum, (a, b) => a + b, 0);
                if (comm.Rank == 0)
                {
                    Console.WriteLine("testing mpi");
                    Console.WriteLine(total);
                }
                //READ DATA

                var frequencies = FitsIO.ReadFrequencies(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\freq.fits");
                var uvw = FitsIO.ReadUVW(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\uvw.fits");
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
                double norm = 2.0;
                var visibilities = FitsIO.ReadVisibilities(@"C:\dev\GitHub\p9-data\small\fits\simulation_point\vis.fits", uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

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

                int gridSize = 128;
                int subgridsize = 16;
                int kernelSize = 8;
                int max_nr_timesteps = 512;
                double cellSize = 2.0 / 3600.0 * PI / 180.0;

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
                var psfCut =  CutImg(psf);
                Complex[,] PsfCorrelation = null;
                if(comm.Rank == 0)
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
                var maxCycle = 1;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    var forwardPass = ForwardCalculateB(comm, c, metadata, residualVis, uvw, frequencies, PsfCorrelation, psfCut, 0.0, watchForward);
                    var bLocal = GetImgSection(forwardPass.Item1, imgSection);
                    if (comm.Rank == 0)
                        watchDeconv.Start();

                    var lambdaStart = 2.5;
                    var lambdaEnd = 0.1;
                    var lambda = lambdaStart - (lambdaStart - lambdaEnd) / (maxCycle) * (cycle + 1);
                    var converged = DistributedGreedyCD.Deconvolve(comm, imgSection, totalImage, xLocal, aMapLocal, bLocal, psfCut, 0.1, 0.8, 1000);
                    
                    //xLocal[0, 0] = 10.0 * (comm.Rank+1);
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
            }
        }

        #region helper methods
        public static double[,] CalculatePSF(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            double[,] psf = null;
            var localGrid = IDG.GridPSF(c, metadata, uvw, flags, frequencies);
            var psf_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                psf = FFT.Backward(psf_total, c.VisibilitiesCount);
                FFT.Shift(psf);    
                Single_Reference.FitsIO.Write(psf, "psf.fits");
                Console.WriteLine("psf Written");
            }
            comm.Broadcast(ref psf, 0);

            return psf;
        }

        public static Tuple<double[,], double> ForwardCalculateB(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, Complex[,] PsfCorrelation, double[,] psfCut, double maxSidelobe, Stopwatch watchIdg)
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
            return new Tuple<double[,], double>(image, maxSideLobeLevel);
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
                var dirtyImage = FFT.Backward(grid_total, c.VisibilitiesCount);
                FFT.Shift(image);
                watchIdg.Stop();

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

        public static double[,] GetImgSection(double[,] b, Common.Rectangle imgSection)
        {
            var yLen = imgSection.YEnd - imgSection.Y;
            var xLen = imgSection.XEnd - imgSection.X;

            var bLocal = new double[yLen, xLen];
            for (int i = 0; i < yLen; i++)
                for (int j = 0; j < xLen; j++)
                    bLocal[i, j] = b[i + imgSection.Y, j + imgSection.X];

            return bLocal;
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
        #endregion


    }
}
