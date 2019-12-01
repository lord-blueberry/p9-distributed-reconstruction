using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;


namespace Core.Deconvolution
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
        DeconvolutionResult DeconvolvePath(float[,] reconstruction, float[,] gradients, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 1e-4f);

        DeconvolutionResult Deconvolve(float[,] reconstruction, float[,] gradients, float lambda, float alpha, int iterations, float epsilon = 1e-4f);

        void ResetLipschitzMap(float[,] psf);

        float GetAbsMaxDiff(float[,] xImage, float[,] gradients, float lambda, float alpha);
    }
}
