using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Core;

namespace SingleReconstruction.ImageGeneration.SerialCD
{
    class Generator
    {
        static string INPUT_DIR = "./ImageGeneration/SerialCD";

        public static void Generate(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var reference = FitsIO.ReadImage(Path.Combine(INPUT_DIR, "xReference4.fits"));

            Tools.WriteToMeltCSV(reference, Path.Combine(outputFolder, "CD-reference.csv"));

            var n132 = Tools.LMC.CutN132Remnant(reference);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "CD-N132.csv"), n132.Item2, n132.Item3);
            var calibration = Tools.LMC.CutCalibration(reference);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "CD-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    }
}
