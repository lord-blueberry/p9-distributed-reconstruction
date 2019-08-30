using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public interface IDeconvolver
    {
        bool DeconvolvePath(float[,] reconstruction, float[,] residuals, float[,] psf, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, double epsilon = 1e-4);
    }
}
