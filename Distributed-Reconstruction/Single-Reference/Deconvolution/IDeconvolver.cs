using System;
using System.Collections.Generic;
using System.Text;
using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public interface IDeconvolver
    {
        bool DeconvolvePath(float[,] reconstruction, float[,] residuals, Rectangle residualsWindow, float[,] psf, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, double epsilon = 1e-4);
    }
}
