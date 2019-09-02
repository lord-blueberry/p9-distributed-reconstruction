using System;
using System.Collections.Generic;
using System.Text;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public interface IDeconvolver
    {
        bool DeconvolvePath(float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 1e-4f);

        bool Deconvolve(float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 1e-4f);
    }
}
