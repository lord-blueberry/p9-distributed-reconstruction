using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;


namespace Single_Reference.Deconvolution
{
    public class DeconvolutionResult
    {
        public bool Converged { get; }
        public int IterationCount { get; }
        public TimeSpan ElapsedTime { get; }

        public DeconvolutionResult(bool converged, int iter, TimeSpan elapsed)
        {
            Converged = converged;
            IterationCount = iter;
            ElapsedTime = elapsed;
        }
    }

    public interface IDeconvolver
    {
        DeconvolutionResult DeconvolvePath(float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 1e-4f);

        DeconvolutionResult Deconvolve(float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 1e-4f);
    }
}
