using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using MPI;
using Single_Reference;

namespace Distributed_Reference
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
            var frequencies = FitsIO.ReadFrequencies(System.IO.Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(System.IO.Path.Combine(folder, "uvw.fits"));
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length]; //completely unflagged dataset
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(System.IO.Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            return new LocalDataset(frequencies, uvw, flags, visibilities);
        }

        public static LocalDataset LoadSubsetTinyMeerKAT(Intracommunicator comm, string folder)
        {
            var frequencies = FitsIO.ReadFrequencies(System.IO.Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(System.IO.Path.Combine(folder, "uvw0.fits"));
            var flags = FitsIO.ReadFlags(System.IO.Path.Combine(folder, "flags0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            var visibilities = FitsIO.ReadVisibilities(System.IO.Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            return new LocalDataset(frequencies, uvw, flags, visibilities);
        }

        public static LocalDataset LoadTinyMeerKAT(Intracommunicator comm, string folder)
        {
            var beginIdx = comm.Rank * 8 / comm.Size;

            var frequencies = FitsIO.ReadFrequencies(System.IO.Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(System.IO.Path.Combine(folder, "uvw" + comm.Rank + ".fits"));
            var flags = FitsIO.ReadFlags(System.IO.Path.Combine(folder, "flags" + comm.Rank + ".fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
            var visibilities = FitsIO.ReadVisibilities(System.IO.Path.Combine(folder, "vis" + comm.Rank + ".fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, 2.0);

            for (int i = beginIdx + 1; i < beginIdx + 8 / comm.Size; i++)
            {
                Console.WriteLine("Rank {0} reads idx {1} ", comm.Rank, i);
                var uvw0 = FitsIO.ReadUVW(System.IO.Path.Combine(folder, "uvw" + i + ".fits"));
                var flags0 = FitsIO.ReadFlags(System.IO.Path.Combine(folder, "flags" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                var visibilities0 = FitsIO.ReadVisibilities(System.IO.Path.Combine(folder, "vis" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, 2.0);
                uvw = FitsIO.Stitch(uvw, uvw0);
                flags = FitsIO.Stitch(flags, flags0);
                visibilities = FitsIO.Stitch(visibilities, visibilities0);
            }

            return new LocalDataset(frequencies, uvw, flags, visibilities);
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
