using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Single_Machine.Deconvolution
{
    /// <summary>
    /// Cooordinate Descent "CLEAN" iteration.
    /// 
    /// It uses Coordinate Descent to minimize the deconvolution of a single pixel.
    /// </summary>
    public class CDClean
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xImage">resulting deconvolved image. If this is the first Major cycle, xImage[i] = 0</param>
        /// <param name="residual">residual image</param>
        /// <param name="psf">point spread function in image space</param>
        /// <param name="lambda">regularization parameter</param>
        public static void CoordinateDescent(double[,] xImage, double[,] residual, double[,] psf, double lambda, int maxIteration = 100)
        {
            int iter = 0;
            bool converged = false;
            var precision = 1e-6;
      
            // the "a" of the parabola equation a* x*x + b*x + c
            var a = CalcPSFSquared(psf);
            while (iter < maxIteration & !converged)
            {
                converged = true;
                iter++;
                for (int y = 0; y < residual.GetLength(0); y++)
                {
                    for (int x = 0; x < residual.GetLength(1); x++)
                    {
                        var xOld = xImage[y, x];
                        var b = CalculateB(residual, psf, y, x);

                        //calculate minimum of parabola, eg -2b/a
                        var xNew = xOld + (b / a);
                        xNew = ShrinkAbsolute(xNew, lambda);
                        var xDiff = xNew - xOld;

                        if (Math.Abs(xDiff) > precision)
                        {
                            converged = false;
                            xImage[y, x] = xNew;
                            ModifyResidual(residual, psf, y, x, xDiff);
                        }
                    }
                }
            }
        }

        private static double CalculateB(double[,] residual, double[,] psf, int y, int x)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;

            var b = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var ySrc = yOffset + i;
                    var xSrc = xOffset + j;
                    if (ySrc >= 0 & ySrc < residual.GetLength(0) &
                        xSrc >= 0 & xSrc < residual.GetLength(1))
                    {
                        b += residual[ySrc, xSrc] * psf[i, j];
                    }
                }
            }

            return b;
        }

        private static void ModifyResidual(double[,] residual, double[,] psf, int y, int x, double xDiff)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var yDst = yOffset + i;
                    var xDst = xOffset + j;
                    if (yDst >= 0 & yDst < residual.GetLength(0) &
                        xDst >= 0 & xDst < residual.GetLength(1))
                    {
                        var res = residual[yDst, xDst];
                        var psff = psf[i, j];
                        residual[yDst, xDst] -= xDiff * psf[i, j];
                    }
                }
            }
        }

        private static double CalcPSFSquared(double[,] psf)
        {
            double squared = 0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(0); x++)
                    squared += psf[y, x] * psf[y, x];
            
            return squared;
        }

        private static double ShrinkAbsolute(double value, double lambda) 
        {
            value = Math.Abs(value) - lambda;
            return Math.Max(value, 0.0);
        }


    }
}
