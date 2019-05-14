using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Single_Reference.Deconvolution
{
    class CyclicCD
    {
        public const double PRECISION = 1e-4;

        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf2, double lambda, double alpha, int maxIteration = 100)
        {
            GreedyCD.Deconvolve(xImage, b, psf2, lambda, alpha, maxIteration);
            var converged = false;
            var iter = 0;
            var aa = psf2[psf2.GetLength(0) / 2, psf2.GetLength(1) / 2];
            while (iter < maxIteration & !converged)
            {
                Console.WriteLine("--------------------adding to active set------------------");
                iter++;
                var activeSet = new HashSet<Tuple<int, int>>();
                for (int i = 0; i < b.GetLength(0); i++)
                    for (int j = 0; j < b.GetLength(1); j++)
                    {
                        var currentB = b[i, j];
                        if (currentB > 0)
                            Console.Write("");
                        var xOld = xImage[i, j];
                        var xDiff = (currentB / aa);

                        var xNew = xOld + xDiff;
                        xNew = xNew / (1 + lambda * (1 - alpha));
                        xNew = Math.Max(xNew, 0.0);
                        //xNew = Math.Max(xNew, 0.0);     //positive
                        if(Math.Abs(xOld - xNew) > PRECISION)
                        {
                            xImage[i, j] = xNew;
                            Console.WriteLine(xDiff + "\t" + i + "\t" + j);
                            activeSet.Add(new Tuple<int, int>(i, j));
                            UpdateB2(b, psf2, i, j, xOld - xNew);
                        }
                    }

                //active set iterations
                converged = activeSet.Count == 0;
                bool activeSetConverged = false;
                var innerMax = 1000;
                var innerIter = 0;
                while (!activeSetConverged & innerIter <= innerMax)
                {
                    activeSetConverged = true;
                    var delete = new List<Tuple<int, int>>();
                    innerIter++;
                    foreach (var pixel in activeSet.ToArray())
                    {
                        var y = pixel.Item1;
                        var x = pixel.Item2;
                        var xOld = xImage[y, x];
                        var currentB = b[y, x];
                        //calculate minimum of parabola, eg -2b/a
                        var xDiff = currentB / aa;
                        var xNew = xOld + xDiff;
                        xNew = xNew / (1 + lambda * (1 - alpha));
                        xNew = Math.Max(xNew, 0.0);
                        if (Math.Abs(xOld - xNew) > PRECISION)
                        {
                            Console.WriteLine(Math.Abs(xOld - xNew) + "\t" + y + "\t" + x);
                            activeSetConverged = false;
                            xImage[y, x] = xNew;
                            UpdateB2(b, psf2, y, x, xOld - xNew);
                        }
                        else if(xNew < PRECISION)
                        {
                            //approximately zero, remove from active set
                            activeSetConverged = false;
                            xImage[y, x] = 0.0;
                            UpdateB2(b, psf2, y, x, xOld);
                            Console.WriteLine("drop pixel \t" + xNew + "\t" + y + "\t" + x);
                            delete.Add(pixel);
                        }
                    }

                    foreach (var pixel in delete)
                        activeSet.Remove(pixel);
                }

            }
            return true;
        }

        private static void UpdateB2(double[,] b, double[,] psf2, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = (int)Math.Ceiling(psf2.GetLength(0) / 2.0);
            var xPsfHalf = (int)Math.Ceiling(psf2.GetLength(1) / 2.0);
            for (int i = 0; i < psf2.GetLength(0); i++)
            {
                for (int j = 0; j < psf2.GetLength(1); j++)
                {
                    var y = (yPixel + i) % b.GetLength(0);
                    var x = (xPixel + j) % b.GetLength(1);
                    var yPsf = (i + yPsfHalf) % psf2.GetLength(0);
                    var xPsf = (j + xPsfHalf) % psf2.GetLength(1);
                    b[y, x] += psf2[yPsf, xPsf] * xDiff;
                }
            }
        }

        public static double CalcPSFSquared(double[,] psf)
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
