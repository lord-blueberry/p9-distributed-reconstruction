using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Single_Reference;
using Single_Reference.IDGSequential;

namespace SingleMachineRuns.ImageGeneration.Simulated
{
    class Generator
    {
        public static void GeneratePSFs(string simulatedLocation, string outputFolder)
        {
            var data = InputData.LoadSimulatedPoints(simulatedLocation);
            var c = InputData.CreateSimulatedStandardParams(data.VisibilitiesCount);
            var metadata = Partitioner.CreatePartition(c, data.UVW, data.Frequencies);

            var psfGrid = IDG.GridPSF(c, metadata, data.UVW, data.Flags, data.Frequencies);
            var psf = FFT.BackwardFloat(psfGrid, c.VisibilitiesCount);
            FFT.Shift(psf);

            Directory.CreateDirectory(outputFolder);

        }
    }
}
