using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using MPI;
using Single_Reference;
using System.IO;

namespace DistributedReconstruction
{
    class DistributedData
    {
        public class LocalDataset
        {
            public double[] Frequencies { get; }
            public double[,,] UVW { get; }
            public bool[,,] Flags { get; }
            public Complex[,,] Visibilities { get; }
            public long VisibilitiesCount { get; }

            public LocalDataset(double[] freq, double[,,] uvw, bool[,,] flags, Complex[,,] vis)
            {
                Frequencies = freq;
                UVW = uvw;
                Flags = flags;
                Visibilities = vis;

                var visCountLocal = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCountLocal++;
                VisibilitiesCount = visCountLocal;
            }
        }

        public static LocalDataset LoadSimulated(string folder)
        {
            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw.fits"));
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            return new LocalDataset(frequencies, uvw, flags, visibilities);
        }

        public static LocalDataset LoadSubsetTinyMeerKAT(string folder)
        {
            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw0.fits"));
            var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            return new LocalDataset(frequencies, uvw, flags, visibilities);
        }

        public static LocalDataset LoadTinyMeerKAT(Intracommunicator comm, string folder)
        {
            var beginIdx = comm.Rank * 8 / comm.Size;

            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + comm.Rank + ".fits"));
            var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags" + comm.Rank + ".fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + comm.Rank + ".fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            for (int i = beginIdx + 1; i < beginIdx + 8 / comm.Size; i++)
            {
                var uvw0 = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + i + ".fits"));
                var flags0 = FitsIO.ReadFlags(Path.Combine(folder, "flags" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, 2.0);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            return new LocalDataset(frequencies, uvw, flags, visibilities);
        }

        public static LocalDataset LoadTinyMeerKAT2(int rank, int nodeCount, string folder)
        {
            var blSum = 0;
            var blFileScans = new int[8];
            var blFileCounts = new int[8];
            for (int i = 0; i < 8; i++)
            {
                blFileCounts[i] = FitsIO.CountBaselines(Path.Combine(folder, "uvw" + i + ".fits"));
                blSum += blFileCounts[i];
                blFileScans[i] = blSum;
            }

            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var blBeginIdx = rank * (int)(blSum / (double)nodeCount);
            var blEndIdx = rank + 1 < nodeCount ? (rank + 1) * (int)(blSum / (double)nodeCount) : blSum;
            var baselineCount = blEndIdx - blBeginIdx;

            LocalDataset output = null;
            for (int i = 0; i < 8; i++)
            {
                if (blBeginIdx < blFileScans[i])
                {
                    var blBefore = i > 0 ? blFileScans[i - 1] : 0;
                    var start = blBeginIdx - blBefore;
                    var end = Math.Min(start + baselineCount, blFileCounts[i]);
                    var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + i + ".fits"), start, end);
                    var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags" + i + ".fits"), start, end, uvw.GetLength(1), frequencies.Length);
                    var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + i + ".fits"), start, end, uvw.GetLength(1), frequencies.Length, 2.0);

                    if (blBeginIdx + baselineCount > blFileCounts[i])
                    {
                        //continue reading files until all baselines, which belong to the current node, are loaded
                        var baselinesLoaded = end - start;
                        for(int j = i+1; j < 8; j++)
                        {
                            var end2 = Math.Min(baselineCount - baselinesLoaded, blFileCounts[j]);
                            var uvw0 = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + j + ".fits"), 0, end2);
                            var flags0 = FitsIO.ReadFlags(Path.Combine(folder, "flags" + j + ".fits"), 0, end2, uvw.GetLength(1), frequencies.Length);
                            var visibilities0 = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + j + ".fits"), 0, end2, uvw.GetLength(1), frequencies.Length, 2.0);

                            uvw = FitsIO.Stitch(uvw, uvw0);
                            flags = FitsIO.Stitch(flags, flags0);
                            visibilities = FitsIO.Stitch(visibilities, visibilities0);
                            baselinesLoaded += end2;
                            if (baselinesLoaded >= baselineCount)
                            {
                                //last file read;
                                break;
                            }
                        }
                    }

                    output = new LocalDataset(frequencies, uvw, flags, visibilities);
                    break;
                }

                
            }

            return output;
        }



        public static LocalDataset SplitDataAmongNodes(Intracommunicator comm, LocalDataset data)
        {
            var nrBaselines = data.UVW.GetLength(0) / comm.Size;
            var nrFrequencies = data.Frequencies.Length;
            var uvwSplit = new double[nrBaselines, data.UVW.GetLength(1), 3];
            var visSplit = new Complex[nrBaselines, data.UVW.GetLength(1), nrFrequencies];
            var flagsSplit = new bool[nrBaselines, data.UVW.GetLength(1), nrFrequencies];
            var blOffset = data.UVW.GetLength(0) / comm.Size * comm.Rank;
            for (int i = 0; i < nrBaselines; i++)
            {
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                {
                    for (int k = 0; k < nrFrequencies; k++)
                    {
                        visSplit[i, j, k] = data.Visibilities[blOffset + i, j, k];
                        flagsSplit[i, j, k] = data.Flags[blOffset + i, j, k];
                    }
                    uvwSplit[i, j, 0] = data.UVW[blOffset + i, j, 0];
                    uvwSplit[i, j, 1] = data.UVW[blOffset + i, j, 1];
                    uvwSplit[i, j, 2] = data.UVW[blOffset + i, j, 2];
                }
            }

            return new LocalDataset(data.Frequencies, uvwSplit, flagsSplit, visSplit);
        }

    }
}
