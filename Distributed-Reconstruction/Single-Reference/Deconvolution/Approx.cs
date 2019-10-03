using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class Approx : IDeconvolver
    {
        private int blockSize;
        private int threadCount;

        public Approx(int blockSize, int threadCount)
        {

        }

        

        #region IDeconvolver implementation
        public DeconvolutionResult Deconvolve(float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 0.0001F)
        {
            throw new NotImplementedException();
        }

        public DeconvolutionResult DeconvolvePath(float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001F)
        {
            throw new NotImplementedException();
        }
        #endregion
    }
}
