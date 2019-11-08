﻿using System;
using System.Collections.Generic;
using System.Text;

namespace SingleMachineRuns.ImageGeneration
{
    class Generator
    {
        /// <summary>
        /// Generate images needed for the documentation
        /// </summary>
        public static void GenerateAll()
        {
            var simulated = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
            var outputFolder = "images/simulated";
            Simulated.Generator.GeneratePSFs(simulated, outputFolder);
        }
    }
}
