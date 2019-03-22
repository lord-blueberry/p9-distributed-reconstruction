using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using System.Numerics;


namespace Single_Machine_Test
{
    [TestClass]
    public class TestSubgridding
    {
        private static double[] frequency = { 857000000f, 859000000f, 861000000f, 863000000f, 865000000f, 867000000f, 869000000f, 871000000f };



        [TestMethod]
        public void SingleVisibility()
        {
            /*  baseline 1036
                timestep 1
                channel 0 */
            double[] frequency = { 857000000f };

            //only xx polarization
            double visR = 3.214400053024292;
            double visI = 0.801982581615448;

            double u = 0.17073525427986169;
            double v = -399.17929423647;
            double w = -2.7543493956327438;


        }
    }
}
