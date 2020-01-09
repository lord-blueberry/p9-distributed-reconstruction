using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Core;

namespace SingleReconstruction.FigureGeneration.SerialCD
{
    class Generator
    {
        static string INPUT_DIR = "./FigureGeneration/SerialCD";

        public static void Generate(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var reference = FitsIO.ReadImage(Path.Combine(INPUT_DIR, "xReference4.fits"));

            Tools.WriteToMeltCSV(reference, Path.Combine(outputFolder, "CD-reference.csv"));

            var n132 = Tools.LMC.CutN132Remnant(reference);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "CD-N132.csv"), n132.Item2, n132.Item3);
            var calibration = Tools.LMC.CutCalibration(reference);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "CD-Calibration.csv"), calibration.Item2, calibration.Item3);

            var residual= FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "cd-residual.fits"));
            Tools.WriteToMeltCSV(residual, Path.Combine(outputFolder, "CD-reference-residuals.csv"));

            var reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "cd-image.fits"));
            var image = new float[reconstruction.GetLength(0), reconstruction.GetLength(1)];
            for (int i = 0; i < reconstruction.GetLength(0); i++)
                for (int j = 0; j < reconstruction.GetLength(1); j++)
                    image[i, j] = reconstruction[i, j] - residual[i, j];

            n132 = Tools.LMC.CutN132Remnant(image);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "CD-image-N132.csv"), n132.Item2, n132.Item3);
            calibration = Tools.LMC.CutCalibration(image);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "CD-image-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    }
}
