using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution.Approx
{
    class SharedData
    {
        public float Lambda { get; set; }
        public float Alpha { get; set; }

        public int ProcessorCount { get; private set; }

        public int DegreeOfSeperability { get; private set; }
        public float[,] Psf2 { get; private set; }
        public float[,] AMap { get; private set; }

        public float[,] XExpl { get; private set; }
        public float[,] XCorr { get; private set; }
        public float[,] GExpl { get; private set; }
        public float[,] GCorr { get; private set; }

        public List<Tuple<int, int>> ActiveSet { get; set; }
        public int[] BlockLock { get; set; }

        public int MaxConcurrentIterations { get; set; }
        public float TestRestart;
        public float Theta0 { get; set; }
        public int AsyncFinished = 0;

        public SharedData(
            float lambda,
            float alpha,
            int processorCount,

            int degreeOfSep,
            float[,] psf2,
            float[,] aMap,

            float[,] xExpl,
            float[,] xCorr,
            float[,] gExpl,
            float[,] gCorr)
        {
            Lambda = lambda;
            Alpha = alpha;
            ProcessorCount = processorCount;

            DegreeOfSeperability = degreeOfSep;
            Psf2 = psf2;
            AMap = aMap;

            XExpl = xExpl;
            XCorr = xCorr;
            GExpl = gExpl;
            GCorr = gCorr;
        }
    }
}
