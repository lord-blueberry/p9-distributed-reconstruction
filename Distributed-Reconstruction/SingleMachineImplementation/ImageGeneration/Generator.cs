using System;
using System.Collections.Generic;
using System.Text;

namespace SingleMachineRuns.ImageGeneration
{
    class Generator
    {
        //generate images needed for the documentation
        public static void GenerateAll()
        {
            var simulated = "";
            var outputFolder = "";
            Simulated.Generator.GeneratePSFs(simulated, outputFolder);
        }
    }
}
