using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class FastGreedyCD : IDeconvolver
    {
        private float[,] aMap;

        public FastGreedyCD(int nCores)
        {

        }

        public bool DeconvolvePath(float[,] reconstruction, float[,] residuals, Common.Rectangle residualsWindow, float[,] psf, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, double epsilon = 0.0001)
        {
            throw new NotImplementedException();
        }
    }
}
