using System;
using System.Collections.Generic;
using System.Text;

namespace SingleReconstruction.Reconstruction
{
    class Start
    {

        public static void StartReconstruction()
        {
            var folder = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var data = MeasurementData.LoadSimulatedPoints(folder);
            var griddingConstants = MeasurementData.CreateSimulatedStandardParams(data.VisibilitiesCount);

            MajorCycle.ReconstructSerialCD(data, griddingConstants, true, 1, 5, 0.5f, 0.2f, 5000, 1e-5f);
            MajorCycle.ReconstructPCDM(data, griddingConstants, 1, 5, 1, 0.5f, 0.2f, 30, 1e-5f);

        }
    }
}
