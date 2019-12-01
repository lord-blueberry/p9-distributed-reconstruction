using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Numerics;
using Core;
using Core.ImageDomainGridder;

namespace SingleMachineRuns
{
    class MeasurementData
    {
        public Complex[,,] Visibilities { get; private set; }
        public double[,,] UVW { get; private set; }
        public double[] Frequencies { get; private set; }
        public bool[,,] Flags { get; private set; }
        public long VisibilitiesCount { get; private set; }

        public MeasurementData(Complex[,,] vis, double[,,] uvw, double[] freq, bool[,,] flags)
        {
            Visibilities = vis;
            UVW = uvw;
            Frequencies = freq;
            Flags = flags;

            VisibilitiesCount = 0;
            for (int i = 0; i < flags.GetLength(0); i++)
                for (int j = 0; j < flags.GetLength(1); j++)
                    for (int k = 0; k < flags.GetLength(2); k++)
                        if (!flags[i, j, k])
                            VisibilitiesCount++;
        }

        public static MeasurementData LoadLMC(string folder)
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

            return new MeasurementData(visibilities, uvw, frequencies, flags);
        }

        public static MeasurementData LoadSimulatedPoints(string folder)
        {
            var frequencies = FitsIO.ReadFrequencies(Path.Combine(folder, "freq.fits"));
            var uvw = FitsIO.ReadUVW(Path.Combine(folder, "uvw.fits"));
            var flags = new bool[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            double norm = 2.0;
            var visibilities = FitsIO.ReadVisibilities(Path.Combine(folder, "vis.fits"), uvw.GetLength(0), uvw.GetLength(1), frequencies.Length, norm);

            return new MeasurementData(visibilities, uvw, frequencies, flags);
        }

        public static GriddingConstants CreateSimulatedStandardParams(long visibilitiesCount)
        {
            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * Math.PI / 180.0;

            return new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
        }

        public static GriddingConstants CreateLMCStandardParams(long visibilitiesCount)
        {
            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * Math.PI / 180.0;

            return new GriddingConstants(visibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
        }


    }
}
