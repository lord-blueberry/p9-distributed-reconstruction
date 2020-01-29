using Core.ImageDomainGridder;
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace SingleReconstruction
{
    class RunningMethod
    {
        const string P9_DATA_FOLDER = "C:/dev/GitHub/Schwammberger-P9-Data/p9-data/";

        public static void StartSimulatedReconstruction()
        {
            var folder = Path.Combine(P9_DATA_FOLDER, "small/fits/simulation_point/");
            var data = MeasurementData.LoadSimulatedPoints(folder);

            int gridSize = 256;
            int subgridsize = 8;
            int kernelSize = 4;
            int max_nr_timesteps = 1024;
            double cellSize = 1.0 / 3600.0 * Math.PI / 180.0;

            var griddingConstants = new GriddingConstants(data.VisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

            var obsName = "simulated";
            var lambda = 0.5f;
            var alpha = 0.2f;
            var epsilon = 1e-5f;
            var maxMajorCycles = 5;
  
            MajorCycle.ReconstructSerialCD(obsName, data, griddingConstants, true, 1, maxMajorCycles, lambda, alpha, 5000, epsilon);
            MajorCycle.ReconstructPCDM(obsName, data, griddingConstants, 1, maxMajorCycles, 1, lambda, alpha, 30, epsilon);
        }

        public static void StartLMCReconstruction()
        {
            var folder = Path.Combine(P9_DATA_FOLDER, "large/fits/meerkat_tiny/");
            var data = MeasurementData.LoadLMC(folder);

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

            var obsName = "LMC";
            var lambda = 1.0f;
            var alpha = 0.01f;
            var epsilon = 1e-5f;
            var maxMajorCycles = 5;

            MajorCycle.ReconstructSerialCD(obsName, data, griddingConstants, true, 16, maxMajorCycles, lambda, alpha, 30000, epsilon);
            MajorCycle.ReconstructPCDM(obsName, data, griddingConstants, 32, maxMajorCycles, 3, lambda, alpha, 30, epsilon);
        }
    }
}
