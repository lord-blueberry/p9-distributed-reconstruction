using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    class GradientDebug
    {
        private static double CalcV(double[,] img)
        {
            var prob = 1.0 / (double)img.Length;
            return (1.0 - prob) / prob;
        }

        public static bool Deconvolve(double[,] xImage, double[,] bMap, double[,] psf, double lambda, double alpha, int maxIteration = 100, double epsilon = 1e-4)
        {
            bool converged = false;
            var aMap = Common.PSF.CalcAMap(xImage, psf);
            var psf2 = Common.PSF.CalcPSFSquared(xImage, psf);
            var xDiff = new double[xImage.GetLength(0), xImage.GetLength(1)];
            var imageSection = new Common.Rectangle(0, 0, xImage.GetLength(0), xImage.GetLength(1));

            var rand = new Random(33);
            var iter = 0;
            var theta = 1; //2; //theta, also number of processors.
            var degreeOfSep = CountNonZero(psf);
            var blockCount = xImage.Length;
            var beta = 1.0 + (degreeOfSep - 1) * (theta - 1) / (Math.Max(1.0, (blockCount - 1))); //arises from E.S.O of theta-nice sampling
                                                                                                  /*
                                                                                                   * Theta-nice sampling := sample theta pixels uniformly at random. I.e. the pixel 
                                                                                                   */

            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var psfTmp = new double[psf.GetLength(0) + +psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
            FFT.Shift(psfTmp);
            var PsfCorr = FFT.Forward(psfTmp, 1.0);

            psfTmp = new double[psf.GetLength(0) + psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
            Common.PSF.SetPSFInWindow(psfTmp, xImage, psf, xImage.GetLength(0) / 2, xImage.GetLength(1) / 2);
            var tmp = FFT.Forward(psfTmp, 1.0);
            var tmp2 = Common.Fourier2D.Multiply(tmp, PsfCorr);



            while (!converged & iter < maxIteration)
            {
                // create sampling
                var samples = CreateSamples(xImage.Length, theta, rand);
                for (int i = 0; i < samples.Length; i++)
                {
                    var yPixel = samples[i] / xImage.GetLength(1);
                    var xPixel = samples[i] % xImage.GetLength(1);

                    //update with E.S.O
                    //var xDiffPixel = 2.0 * bMap[y, x] / (beta * aMap[y, x]);

                    //update without E.S.O
                    var xDiffPixel = bMap[yPixel, xPixel] / (aMap[yPixel, xPixel]);

                    var old = xImage[yPixel, xPixel];
                    var xDiffShrink = Common.ShrinkElasticNet(old + xDiffPixel, lambda, alpha);
                    xDiff[yPixel, xPixel] = xDiffShrink;
                }

                //update B-map
                //reset xDiff
                for (int i = 0; i < samples.Length; i++)
                {
                    var yPixel = samples[i] / xImage.GetLength(1);
                    var xPixel = samples[i] % xImage.GetLength(1);

                    var bla = xDiff[yPixel, xPixel];
                    if( bla != 0.0)
                    {
                        if (yPixel - yPsfHalf >= 0 & yPixel + yPsfHalf < xImage.GetLength(0) & xPixel - xPsfHalf >= 0 & xPixel + xPsfHalf < xImage.GetLength(0))
                        {
                            UpdateB(bMap, psf2, imageSection, yPixel, xPixel, -xDiff[yPixel, xPixel]);
                        }
                        else
                        {
                            Common.PSF.SetPSFInWindow(psfTmp, xImage, psf, yPixel, xPixel);
                            tmp = FFT.Forward(psfTmp, 1.0);
                            tmp2 = Common.Fourier2D.Multiply(tmp, PsfCorr);
                            var bUpdateMasked = FFT.Backward(tmp2, tmp2.GetLength(0) * tmp2.GetLength(1));
                            UpdateB(bMap, bUpdateMasked, imageSection, yPixel, xPixel, -xDiff[yPixel, xPixel]);
                        }

                        xImage[yPixel, xPixel] += xDiff[yPixel, xPixel];
                        xDiff[yPixel, xPixel] = 0;
                        Console.WriteLine("iter " + iter + " y " + yPixel + " x " + xPixel + " val " + bla);
                    }

                    Console.WriteLine("iter " + iter);


                }
                
                iter++;
                
            }

            return converged;
        }

        private static void UpdateB(double[,] b, double[,] bUpdate, Common.Rectangle imageSection, int yPixel, int xPixel, double xDiff)
        {
            var yBHalf = bUpdate.GetLength(0) / 2;
            var xBHalf = bUpdate.GetLength(1) / 2;

            var yBMin = Math.Max(yPixel - yBHalf, imageSection.Y);
            var xBMin = Math.Max(xPixel - xBHalf, imageSection.X);
            var yBMax = Math.Min(yPixel - yBHalf + bUpdate.GetLength(0), imageSection.YEnd);
            var xBMax = Math.Min(xPixel - xBHalf + bUpdate.GetLength(1), imageSection.XEnd);
            for (int i = yBMin; i < yBMax; i++)
                for (int j = xBMin; j < xBMax; j++)
                {
                    var yLocal = i - imageSection.Y;
                    var xLocal = j - imageSection.X;
                    var yBUpdate = i + yBHalf - yPixel;
                    var xBUpdate = j + xBHalf - xPixel;
                    b[yLocal, xLocal] += bUpdate[yBUpdate, xBUpdate] * xDiff;
                }
        }

        private static int[] CreateSamples(int length, int sampleCount, Random rand)
        {
            var samples = new HashSet<int>(sampleCount);
            while(samples.Count < sampleCount)
                samples.Add(rand.Next(0, length));
            int[] output = new int[samples.Count];
            samples.CopyTo(output);
            return output;
        }

        private static int CountNonZero(double[,] psf)
        {
            var count = 0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    if (psf[y, x] != 0.0)
                        count++;
            return count;
        }

        #region old stuff
        public static void Run()
        {
            var psf = new double[8, 8];
            for (int i = 3; i < 6; i++)
                for (int j = 3; j < 6; j++)
                    psf[i, j] = 0.5;
            psf[4, 4] = 1.0;

            var dirty = new double[32, 32];
            var x = new double[32, 32];
            SetPsf(dirty, psf);

            int yi, xi;
            yi = xi = 16;

            var g0 = CalcGradient(dirty, psf, 15, 15);
            var g1 = CalcGradient(dirty, psf, 14, 14);
            var gradient = CalcGradient(dirty, psf, yi, xi);

            var update = 0.5 * gradient;
            x[yi, xi] -= update;

        }

        private static void SetPsf(double[,] image, double[,] psf)
        {
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                    image[i + 16 - 4, j + 16 - 4] = 2.0 * psf[i, j];
        }

        private static double CalcGradient(double[,] residuals, double[,] psf, int y, int x)
        {
            var gradient = 0.0;
            var psfSum = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    gradient += residuals[y + i - 4, x + j - 4] * psf[i, j];
                    psfSum += psf[i, j];
                }


            return -2.0 * gradient / psf.Length;
        }
        #endregion
    }
}
