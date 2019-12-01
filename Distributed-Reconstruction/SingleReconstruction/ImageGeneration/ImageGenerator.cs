using System;
using System.Collections.Generic;
using System.Text;

namespace SingleReconstruction.ImageGeneration
{
    class ImageGenerator
    {
        /// <summary>
        /// Generate images needed for the documentation
        /// </summary>
        public static void GenerateAll()
        {
            SerialCD.Generator.Generate("images/SerialCD");
            MSClean.Generator.Generate("images/MSClean");
            var simulated = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var outputFolder = "images/simulated";
            Simulated.Generator.GeneratePSFs(simulated, outputFolder);
            Simulated.Generator.GenerateCLEANExample(simulated, outputFolder);

           
        }
    }
}
