using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Single_Reference.Deconvolution
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
        public static void Deconvolve(double[,] xImage, double[,] residual, double[,] psf, double lambda, int maxIteration = 100)
        {
            int iter = 0;
            bool converged = false;
            var precision = 1e-4;
      
            // the "a" of the parabola equation a* x*x + b*x + c
            var a = CalcPSFSquared(psf);
            while (iter < maxIteration & !converged)
            {
                iter++;
                var activeSet = new HashSet<Tuple<int, int>>();
                
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
                            activeSet.Add(new Tuple<int, int>(y, x));
                            xImage[y, x] = xNew;
                            ModifyResidual(residual, psf, y, x, xDiff);
                        }
                    }
                }


                converged = activeSet.Count  == 0;
                bool activeSetConverged = false;
                var innerMax = 1000;
                var innerIter = 0;
                while (!activeSetConverged | innerIter <= innerMax)
                {
                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    innerIter++;
                    foreach (var pixel in activeSet)
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var xOld = xImage[y, x];
                        var b = CalculateB(residual, psf, y, x);

                        //calculate minimum of parabola, eg -2b/a
                        var xNew = xOld + (b / a);
                        xNew = ShrinkAbsolute(xNew, lambda);
                        var xDiff = xNew - xOld;

                        //approximately zero, remove from active set
                        if (xNew == 0.0)
                        {
                            activeSetConverged = false;
                            xImage[y, x] = 0.0;
                            ModifyResidual(residual, psf, y, x, xDiff);
                            delete.Add(pixel);
                        }
                        else if (Math.Abs(xDiff) > precision)
                        {
                            activeSetConverged = false;
                            xImage[y, x] = xNew;
                            ModifyResidual(residual, psf, y, x, xDiff);
                        }
                    }

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }
                //converged = true;
            }
        }

        private static double CalculateB(double[,] residual, double[,] psf, int y, int x)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;

            var b = 0.0;            
            for(int i= 0; i <  psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var ySrc = Math.Abs(yOffset + i);
                    var xSrc = Math.Abs(xOffset + j);
                    ySrc = ySrc >= residual.GetLength(0) ? 2 * (residual.GetLength(0) - 1) - ySrc : ySrc;
                    xSrc = xSrc >= residual.GetLength(1) ? 2 * (residual.GetLength(1) - 1) - xSrc : xSrc;

                    b += residual[ySrc, xSrc] * psf[i, j];
                }
            }

            return b;
        }

        private static double CalculateBParallel(double[,] residual, double[,] psf, int y, int x)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;

            var lockObj = new Object();
            var b2 = 0.0;

            Parallel.For(0, psf.GetLength(0), () => 0.0, (i, loop, b2Local) =>
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var ySrc = yOffset + i;
                    var xSrc = xOffset + j;
                    if (ySrc >= 0 & ySrc < residual.GetLength(0) &
                        xSrc >= 0 & xSrc < residual.GetLength(1))
                    {
                        b2Local += residual[ySrc, xSrc] * psf[i, j];
                    }
                }
                return b2Local;
            }, (local) =>
            {
                lock (lockObj)
                {
                    b2 += local;
                }
            });

            return b2;
        }


        private static void ModifyResidual(double[,] residual, double[,] psf, int y, int x, double xDiff)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;
            for(int i = 0; i < psf.GetLength(0);  i++ )
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var yDst = yOffset + i;
                    var xDst = xOffset + j;
                    if (yDst >= 0 & yDst < residual.GetLength(0) &
                        xDst >= 0 & xDst < residual.GetLength(1))
                    {
                        var res = residual[yDst, xDst];
                        residual[yDst, xDst] -= xDiff * psf[i, j];
                    }
                }
            }
        }

        private static void ModifyResidualParallel(double[,] residual, double[,] psf, int y, int x, double xDiff)
        {
            int yOffset = y - psf.GetLength(0) / 2;
            int xOffset = x - psf.GetLength(1) / 2;
            Parallel.For(0, psf.GetLength(0), i =>
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var yDst = yOffset + i;
                    var xDst = xOffset + j;
                    if (yDst >= 0 & yDst < residual.GetLength(0) &
                        xDst >= 0 & xDst < residual.GetLength(1))
                    {
                        var res = residual[yDst, xDst];
                        residual[yDst, xDst] -= xDiff * psf[i, j];
                    }
                }
            });
            
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
            value = Math.Max(value, 0.0) - lambda;
            return Math.Max(value, 0.0);
        }


    }
}
