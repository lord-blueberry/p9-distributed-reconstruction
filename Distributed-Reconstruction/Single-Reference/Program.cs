﻿using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            Experiments.PSFSize.Run();
            //Experiments.PSFSize.DebugConvergence();
            
            //Deconvolution.ToyImplementations.RandomBlockCD2.RunToy();
            //DebugMethods.DebugSimulatedGreedy2();

            //DebugMethods.DebugSimulatedPCDM();
            //DebugMethods.DebugILGPU();

            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            //DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

    }
}
