using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            var bMap = new float[16, 16];
            var bMap2 = new float[16, 16];
            var psf2 = new float[8, 8];
            psf2[4, 4] = 1.0f; psf2[4, 5] = 0.5f; psf2[4, 6] = 0.5f;
            var blocks = new float[1, 2, 2];
            blocks[0, 0, 0] = 1.0f;
            //blocks[0, 0, 1] = 2.0f;
            //blocks[0, 1, 0] = 3.0f;
            //blocks[0, 1, 1] = 4.0f;
            


            Single_Reference.Deconvolution.ToyImplementations.ApproxSingle.UpdateBMaps(0, blocks, 0, 0, psf2, bMap, bMap2, 3.0f);

            FitsIO.Write(bMap, "bMapCray.fits");
            //Experiments.PSFSize.Run();
            //Experiments.PSFSize.DebugConvergence();
            //Experiments.PSFSize.DebugConvergence2();

            //Deconvolution.ToyImplementations.RandomBlockCD2.RunToy();
            //DebugMethods.DebugSimulatedGreedy2();

            DebugMethods.DebugSimulatedPCDM();
            //DebugMethods.DebugILGPU();

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

    }
}
