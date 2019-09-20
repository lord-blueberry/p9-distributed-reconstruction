using System;
using System.Collections.Generic;
using System.Text;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    /// <summary>
    /// Interface for subpatch deconvolver. These methods assume that reconstruction[,] and bMap[,] do not contain the whole image, but merely a patch of the whole image. The Deconvolution will only take place in a subpatch of reconstruction[,] and bMap[,].
    /// </summary>
    public interface ISubpatchDeconvolver
    {
        DeconvolutionResult DeconvolvePath(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 1e-4f);

        DeconvolutionResult Deconvolve(Rectangle subpatch, float[,] reconstruction, float[,] bMap, float lambda, float alpha, int iterations, float epsilon = 1e-4f);
    }
}
