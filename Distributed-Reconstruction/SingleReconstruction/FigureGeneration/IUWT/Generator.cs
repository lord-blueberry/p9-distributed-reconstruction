using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Core;

namespace SingleReconstruction.FigureGeneration.IUWT
{
    class Generator
    {
        static string INPUT_DIR = "./FigureGeneration/IUWT";

        public static void Generate(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var model = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-iuwt-model.fits"));
            var residual = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-iuwt-residual.fits"));
            Tools.WriteToMeltCSV(residual, Path.Combine(outputFolder, "iuwt-residuals.csv"));

            Tools.WriteToMeltCSV(model, Path.Combine(outputFolder, "iuwt-model.csv"));

            var reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-iuwt-image.fits"));
            var image = new float[reconstruction.GetLength(0), reconstruction.GetLength(1)];
            for (int i = 0; i < reconstruction.GetLength(0); i++)
                for (int j = 0; j < reconstruction.GetLength(1); j++)
                    image[i, j] = reconstruction[i, j] - residual[i, j];

            var n132 = Tools.LMC.CutN132Remnant(image);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "iuwt-image-N132.csv"), n132.Item2, n132.Item3);
            var calibration = Tools.LMC.CutCalibration(image);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "iuwt-image-Calibration.csv"), calibration.Item2, calibration.Item3);

            n132 = Tools.LMC.CutN132Remnant(model);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "iuwt-N132.csv"), n132.Item2, n132.Item3);
            calibration = Tools.LMC.CutCalibration(model);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "iuwt-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    
    }
}
