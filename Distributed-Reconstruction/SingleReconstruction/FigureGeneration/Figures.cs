using System;
using System.Collections.Generic;
using System.Text;

namespace SingleReconstruction.FigureGeneration
{
    class Figures
    {
        /// <summary>
        /// Generate images needed for the documentation
        /// </summary>
        public static void GenerateAll()
        {
            //ApproxImages.Generator.Generate("images/ApproxImages");

            MSClean.Generator.Generate("images/MSClean");
            SerialCD.Generator.Generate("images/SerialCD");
            IUWT.Generator.Generate("images/IUWT");
            
            var simulated = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var outputFolder = "images/simulated";
            //Simulated.Generator.GeneratePSFs(simulated, outputFolder);
            //Simulated.Generator.GenerateCLEANExample(simulated, outputFolder);
        }
    }
}
