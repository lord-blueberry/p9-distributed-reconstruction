using System;
using System.Collections.Generic;
using System.Text;
using Core.ImageDomainGridder;
using System.IO;
using System.Numerics;
using Core;

namespace SingleReconstruction.Experiments
{
    static class DataLoading
    {
        public class Data
        {
            public GriddingConstants c;
            public List<List<Subgrid>> metadata;
            public double[] frequencies;
            public Complex[,,] visibilities;
            public double[,,] uvw;
            public bool[,,] flags;
            public long visibilitiesCount;
        }

        public static class LMC
        {
            public static Data Load(string folder)
            {
                var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
                var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw0.fits"));
                var flags = FitsIO.ReadFlags(Path.Combine(folder, "flags0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length);
                double norm = 2.0;
                var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis0.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

                for (int i = 1; i < 8; i++)
                {
                    var uvw0 = FitsIO.ReadUVW(Path.Combine(folder, "uvw" + i + ".fits"));
                    var flags0 = FitsIO.ReadFlags(Path.Combine(folder, "flags" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length);
                    var visibilities0 = FitsIO.ReadVisibilities(Path.Combine(folder, "vis" + i + ".fits"), uvw0.GetLength(0), uvw0.GetLength(1), frequencies.Length, norm);
                    uvw = FitsIO.Stitch(uvw, uvw0);
                    flags = FitsIO.Stitch(flags, flags0);
                    visibilities = FitsIO.Stitch(visibilities, visibilities0);
                }

                var visCount2 = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCount2++;
                var visibilitiesCount = visCount2;

                var d = new Data();
                d.frequencies = frequencies;
                d.visibilities = visibilities;
                d.uvw = uvw;
                d.flags = flags;
                d.visibilitiesCount = visibilitiesCount;

                return d;
            }

            public static Data LoadWithStandardParams(string folder)
            {
                var d = Load(folder);
                int gridSize = 2048;
                int subgridsize = 16;
                int kernelSize = 4;
                //cell = image / grid
                int max_nr_timesteps = 512;
                double scaleArcSec = 2.5 / 3600.0 * Math.PI / 180.0;

                d.c = new GriddingConstants(d.visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)scaleArcSec, 1, 0.0f);
                d.metadata = Partitioner.CreatePartition(d.c, d.uvw, d.frequencies);

                return d;
            }
        }

        public static class SimulatedPoints
        {
            public static Data Load(string folder)
            {
                var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
                var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw.fits"));
                var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
                double norm = 2.0;
                var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

                var visCount2 = 0;
                for (int i = 0; i < flags.GetLength(0); i++)
                    for (int j = 0; j < flags.GetLength(1); j++)
                        for (int k = 0; k < flags.GetLength(2); k++)
                            if (!flags[i, j, k])
                                visCount2++;
                var visibilitiesCount = visCount2;

                var d = new Data();
                d.frequencies = frequencies;
                d.visibilities = visibilities;
                d.uvw = uvw;
                d.flags = flags;
                d.visibilitiesCount = visibilitiesCount;

                return d;
            }
        }
    }
}
