﻿using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Core;

namespace SingleReconstruction.FigureGeneration.MSClean
{
    class Generator
    {
        static string INPUT_DIR = "./FigureGeneration/MSClean";

        public static void Generate(string outputFolder)
        {
            Directory.CreateDirectory(outputFolder);
            var reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-briggs-MFS-image.fits"));
            var residual = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "img-comparison-briggs-MFS-residual.fits"));
            Tools.WriteToMeltCSV(residual, Path.Combine(outputFolder, "briggs-CLEAN-residuals.csv"));
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


            var psf = Common.PSF.Cut(FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-clean-psf.fits")), 4);
            FitsIO.Write(psf, Path.Combine(outputFolder, "natclean-PSF.fits"));
            Tools.WriteToMeltCSV(psf, Path.Combine(outputFolder, "natclean-PSF.csv"), psf.GetLength(0), psf.GetLength(1));


            reconstruction = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-clean-image.fits"));
            residual = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-clean-residual.fits"));
            image = new float[reconstruction.GetLength(0), reconstruction.GetLength(1)];
            for (int i = 0; i < reconstruction.GetLength(0); i++)
                for (int j = 0; j < reconstruction.GetLength(1); j++)
                    image[i, j] = reconstruction[i, j] - residual[i, j];

            var example = Tools.LMC.CutExampleImage(image);
            Tools.WriteToMeltCSV(example.Item1, Path.Combine(outputFolder, "natclean-example.csv"), example.Item2, example.Item3);

            FitsIO.Write(image, Path.Combine(outputFolder, "NaturalMSClean.fits"));
            var model = FitsIO.ReadCASAFits(Path.Combine(INPUT_DIR, "nat-clean-model.fits"));
            Tools.WriteToMeltCSV(model, Path.Combine(outputFolder, "Natural-CLEAN.csv"));
            Tools.WriteToMeltCSV(residual, Path.Combine(outputFolder, "Natural-CLEAN-residuals.csv"));

            n132 = Tools.LMC.CutN132Remnant(model);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "Natural-N132.csv"), n132.Item2, n132.Item3);
            calibration = Tools.LMC.CutCalibration(model);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "Natural-Calibration.csv"), calibration.Item2, calibration.Item3);

            n132 = Tools.LMC.CutN132Remnant(image);
            Tools.WriteToMeltCSV(n132.Item1, Path.Combine(outputFolder, "Natural-image-N132.csv"), n132.Item2, n132.Item3);
            calibration = Tools.LMC.CutCalibration(image);
            Tools.WriteToMeltCSV(calibration.Item1, Path.Combine(outputFolder, "Natural-image-Calibration.csv"), calibration.Item2, calibration.Item3);
        }
    }
}
