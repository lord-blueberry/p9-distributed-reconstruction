using Core.ImageDomainGridder;
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace SingleReconstruction
{
    class RunningMethod
    {
        const string P9_DATA_FOLDER = @"C:\dev\GitHub\p9-data";

        public static void StartSimulatedReconstruction()
        {
            var folder = Path.Combine(P9_DATA_FOLDER, @"\small\fits\simulation_point\");
            var data = MeasurementData.LoadSimulatedPoints(folder);

            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * Math.PI / 180.0;

            var griddingConstants = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            MajorCycle.ReconstructSerialCD(data, griddingConstants, true, 1, 5, 0.5f, 0.2f, 5000, 1e-5f);
            MajorCycle.ReconstructPCDM(data, griddingConstants, 1, 5, 1, 0.5f, 0.2f, 30, 1e-5f);

        }

        public static void StartLMCReconstruction()
        {
            var folder = Path.Combine(P9_DATA_FOLDER, @"\large\fits\meerkat_tiny\");
            var data = MeasurementData.LoadSimulatedPoints(folder);

            int gridSize = 3072;
            int subgridsize = 32;
            int kernelSize = 16;
            int max_nr_timesteps = 1024;
            double cellSize = 1.5 / 3600.0 * Math.PI / 180.0;
            int wLayerCount = 24;

            var maxW = 0.0;
            for (int i = 0; i < data.UVW.GetLength(0); i++)
                for (int j = 0; j < data.UVW.GetLength(1); j++)
                    maxW = Math.Max(maxW, Math.Abs(data.UVW[i, j, 2]));
            maxW = Partitioner.MetersToLambda(maxW, data.Frequencies[data.Frequencies.Length - 1]);
            double wStep = maxW / (wLayerCount);

            var griddingConstants = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, wLayerCount, wStep);

            MajorCycle.ReconstructSerialCD(data, griddingConstants, true, 16, 5, 1.0f, 0.01f, 30000, 1e-5f);
            MajorCycle.ReconstructPCDM(data, griddingConstants, 32, 5, 3, 1.0f, 0.01f, 30, 1e-5f);
        }
    }
}
