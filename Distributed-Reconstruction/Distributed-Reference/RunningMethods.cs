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
    class RunningMethods
    {
        public static void RunTinyMeerKAT(string[] args)
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
                //int gridSize = 128;
                int subgridsize = 32;
                int kernelSize = 16;
                int max_nr_timesteps = 512;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;
                //double cellSize = 2.0 / 3600.0 * PI / 180.0;

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
                var psfCut = CalculatePSF(comm, c, metadata, uvw, flags, frequencies);
                Complex[,] PsfCorrelation = null;
                if (comm.Rank == 0)
                    PsfCorrelation = GreedyCD2.PadAndInvertPsf(psfCut, c.GridSize, c.GridSize);

                var halfComm = comm.Size / 2;
                var yResOffset = comm.Rank / 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var imgSection = new Communication.Rectangle(yResOffset, xResOffset, yResOffset + gridSize / halfComm, xResOffset + gridSize / halfComm);
                var totalImage = new Communication.Rectangle(0, 0, c.GridSize, c.GridSize);

                var residualVis = visibilities;
                var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
                var maxCycle = 5;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    var b = ForwardCalculateB(comm, c, metadata, residualVis, uvw, frequencies, PsfCorrelation, psfCut, watchForward);
                    var bLocal = GetImgSection(b, imgSection);
                    var bCopy = new double[c.GridSize, c.GridSize];
                    for (int i = 0; i < c.GridSize; i++)
                        for (int j = 0; j < c.GridSize; j++)
                            bCopy[i, j] = b[i, j];
                    if (comm.Rank == 0)
                        watchDeconv.Start();

                    var lambdaStart = 2.5;
                    var lambdaEnd = 0.1;
                    var lambda = lambdaStart - (lambdaStart - lambdaEnd) / (maxCycle) * (cycle + 1);
                    var converged = DGreedyCD2.Deconvolve(comm, imgSection, totalImage, xLocal, bLocal, psfCut, lambda, 0.4, 10000);
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
                        modelGrid = FFT.GridFFT(x);
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
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);
                var psfCut = CalculatePSF(comm, c, metadata, uvw, flags, frequencies);
                Complex[,] PsfCorrelation = null;
                if(comm.Rank == 0)
                    PsfCorrelation = GreedyCD2.PadAndInvertPsf(psfCut, c.GridSize, c.GridSize);

                var halfComm = comm.Size / 2;
                var yResOffset = comm.Rank / 2 * (gridSize / halfComm);
                var xResOffset = comm.Rank % 2 * (gridSize / halfComm);
                var imgSection = new Communication.Rectangle(yResOffset, xResOffset, yResOffset + gridSize / halfComm, xResOffset + gridSize / halfComm);
                var totalImage = new Communication.Rectangle(0, 0, c.GridSize, c.GridSize);

                var residualVis = visibilities;
                var xLocal = new double[c.GridSize / halfComm, c.GridSize / halfComm];
                var maxCycle = 1;
                for (int cycle = 0; cycle < maxCycle; cycle++)
                {
                    var b = ForwardCalculateB(comm, c, metadata, residualVis, uvw, frequencies, PsfCorrelation, psfCut, watchForward);
                    var bLocal = GetImgSection(b, imgSection);
                    var bCopy = new double[c.GridSize, c.GridSize];
                    for (int i = 0; i < c.GridSize; i++)
                        for (int j = 0; j < c.GridSize; j++)
                            bCopy[i, j] = b[i, j];
                    if (comm.Rank == 0)
                        watchDeconv.Start();

                    var lambdaStart = 2.5;
                    var lambdaEnd = 0.1;
                    var lambda = lambdaStart - (lambdaStart - lambdaEnd) / (maxCycle) * (cycle + 1);
                    var converged = DGreedyCD2.Deconvolve(comm, imgSection, totalImage, xLocal, bLocal, psfCut, 0.1, 0.8, 1000);
                    
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
                        modelGrid = FFT.GridFFT(x);
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
                psf = FFT.GridIFFT(psf_total, c.VisibilitiesCount);
                FFT.Shift(psf);
                psf = CutImg(psf);
                Single_Reference.FitsIO.Write(psf, "psf.fits");
                Console.WriteLine("psf Written");
            }
            comm.Broadcast(ref psf, 0);

            return psf;
        }

        public static double[,] ForwardCalculateB(Intracommunicator comm, GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, Complex[,] PsfCorrelation, double[,] psfCut, Stopwatch watchIdg)
        {
            Stopwatch another = new Stopwatch();
            comm.Barrier();
            if (comm.Rank == 0)
            {
                watchIdg.Start();
            }
                
            
            var localGrid = IDG.Grid(c, metadata, visibilities, uvw, frequencies);
            double[,] image = null;
            if(comm.Rank == 0)
                another.Start();
            var grid_total = comm.Reduce<Complex[,]>(localGrid, SequentialSum, 0);
            if (comm.Rank == 0)
            {
                another.Stop();
                var dirtyImage = FFT.GridIFFT(grid_total, c.VisibilitiesCount);
                FFT.Shift(dirtyImage);
                
                //remove spheroidal
                var dirtyPadded = GreedyCD2.PadResiduals(dirtyImage, psfCut);
                var DirtyPadded = FFT.FFTDebug(dirtyPadded, 1.0);
                var B = IDG.Multiply(DirtyPadded, PsfCorrelation);
                var bPadded = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                image = GreedyCD2.RemovePadding(bPadded, psfCut);
                watchIdg.Stop();
                Console.WriteLine("communication time: " + another.Elapsed);
            }

            comm.Broadcast<double[,]>(ref image, 0);
            return image;
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
                var dirtyImage = FFT.GridIFFT(grid_total, c.VisibilitiesCount);
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

        public static double[,] GetImgSection(double[,] b, Communication.Rectangle imgSection)
        {
            var yLen = imgSection.YLength - imgSection.Y;
            var xLen = imgSection.XLength - imgSection.X;

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
