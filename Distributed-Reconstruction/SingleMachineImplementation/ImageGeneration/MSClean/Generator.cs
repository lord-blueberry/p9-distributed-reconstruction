﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Single_Reference;

namespace SingleMachineRuns.ImageGeneration.MSClean
{
    class Generator
    {
        static string INPUT_DIR = "./ImageGeneration/MSClean";

        public static void Generate(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-briggs-MFS-image.fits"));
            var residual = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-briggs-MFS-residual.fits"));
            var image = new float[reconstruction.GetLength(0), reconstruction.GetLength(1)];
            for (int i = 0; i < reconstruction.GetLength(0); i++)
                for (int j = 0; j < reconstruction.GetLength(1); j++)
                    image[i, j] = reconstruction[i, j] - residual[i, j];
            FitsIO.Write(image, Path.Combine(outputFolder, "MSClean.fits"));
            Tools.WriteToMeltCSV(image, Path.Combine(outputFolder, "Briggs-CLEAN.csv"));

            var n132 = Tools.LMC.CutN132Remnant(image);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "Briggs-N132.csv"), n132.Item2, n132.Item3);
            var calibration = Tools.LMC.CutCalibration(image);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "Briggs-Calibration.csv"), calibration.Item2, calibration.Item3);


            reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-MFS-image.fits"));
            residual = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-MFS-residual.fits"));
            image = new float[reconstruction.GetLength(0), reconstruction.GetLength(1)];
            for (int i = 0; i < reconstruction.GetLength(0); i++)
                for (int j = 0; j < reconstruction.GetLength(1); j++)
                    image[i, j] = reconstruction[i, j] - residual[i, j];
            FitsIO.Write(image, Path.Combine(outputFolder, "NaturalMSClean.fits"));
            Tools.WriteToMeltCSV(image, Path.Combine(outputFolder, "Natural-CLEAN.csv"));

            n132 = Tools.LMC.CutN132Remnant(image);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "Natural-N132.csv"), n132.Item2, n132.Item3);
            calibration = Tools.LMC.CutCalibration(image);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "Natural-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    }
}