using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;

namespace Single_Reference
{
    public static class Common
    {
        public static double ShrinkElasticNet(double value, double lambda, double alpha) => Math.Max(value - lambda * alpha, 0.0) / (1 + lambda * (1 - alpha));
        public static float ShrinkElasticNet(float value, float lambda, float alpha) => Math.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        public static class PSF
        {
            private static float[,] CalcPSFScan(float[,] psf)
            {
                var scan = new float[psf.GetLength(0), psf.GetLength(1)];
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                    {
                        var iBefore = i > 0 ? scan[i - 1, j] : 0.0f;
                        var jBefore = j > 0 ? scan[i, j - 1] : 0.0f;
                        var ijBefore = i > 0 & j > 0 ? scan[i - 1, j - 1] : 0.0f;
                        var current = psf[i, j] * psf[i, j];
                        scan[i, j] = current + iBefore + jBefore - ijBefore;
                    }

                return scan;
            }

            private static float QueryScan(float[,] psfScan, int yPixel, int xPixel, int yLength, int xLength)
            {
                var yOverShoot = (yPixel + (psfScan.GetLength(0) - psfScan.GetLength(0) / 2)) - yLength;
                var xOverShoot = (xPixel + (psfScan.GetLength(1) - psfScan.GetLength(1) / 2)) - xLength;
                yOverShoot = Math.Max(0, yOverShoot);
                xOverShoot = Math.Max(0, xOverShoot);
                var yUnderShoot = (-1) * (yPixel - psfScan.GetLength(0) / 2);
                var xUnderShoot = (-1) * (xPixel - psfScan.GetLength(1) / 2);
                var yUnderShootIdx = Math.Max(1, yUnderShoot) - 1;
                var xUnderShootIdx = Math.Max(1, xUnderShoot) - 1;

                //PSF completely in picture
                if (yOverShoot <= 0 & xOverShoot <= 0 & yUnderShoot <= 0 & xUnderShoot <= 0)
                    return psfScan[psfScan.GetLength(0) - 1, psfScan.GetLength(1) - 1];

                if (yUnderShoot > 0 & xUnderShoot > 0)
                    return psfScan[psfScan.GetLength(0) - 1, psfScan.GetLength(1) - 1]
                           - psfScan[psfScan.GetLength(0) - 1, xUnderShootIdx]
                           - psfScan[yUnderShootIdx, psfScan.GetLength(1) - 1]
                           + psfScan[yUnderShootIdx, xUnderShootIdx];

                var correction = 0.0f;
                if (yUnderShoot > 0)
                    correction += psfScan[yUnderShootIdx, psfScan.GetLength(1) - xOverShoot - 1];
                if (xUnderShoot > 0)
                    correction += psfScan[psfScan.GetLength(0) - yOverShoot - 1, xUnderShootIdx];

                return psfScan[psfScan.GetLength(0) - 1 - yOverShoot, psfScan.GetLength(1) - 1 - xOverShoot] - correction;
            }

            public static float[,] CalcAMap(float[,] psf, Rectangle totalSize, Rectangle imageSection)
            {
                //TODO: change this?
                var scan = CalcPSFScan(psf);
                var aMap = new float[imageSection.YExtent(), imageSection.XExtent()];
                for (int y = imageSection.Y; y < imageSection.YEnd; y++)
                    for (int x = imageSection.X; x < imageSection.XEnd; x++)
                    {
                        var yLocal = y - imageSection.Y;
                        var xLocal = x - imageSection.X;
                        aMap[yLocal, xLocal] = QueryScan(scan, y, x, totalSize.YEnd, totalSize.XEnd);
                    }

                return aMap;
            }

            /// <summary>
            /// invert PSF to calculate the CORRELATION in fourier space (multiplication in fourier space == convolution, multiplication with inverted kernel in fourier space == correlation)
            /// </summary>
            /// <param name="psf"></param>
            /// <param name="padding"></param>
            /// <returns></returns>
            public static Complex[,] CalcPaddedFourierCorrelation(float[,] psf, Rectangle padding)
            {
                var yPadding = padding.YEnd;
                var xPadding = padding.XEnd;

                var psfPadded = new double[yPadding + psf.GetLength(0), xPadding + psf.GetLength(1)];
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                for (int y = 0; y < psf.GetLength(0); y++)
                    for (int x = 0; x < psf.GetLength(1); x++)
                        if (y + +yPsfHalf + 1 < psfPadded.GetLength(0) & x + xPsfHalf + 1 < psfPadded.GetLength(1))
                            psfPadded[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
                FFT.Shift(psfPadded);
                var PSFPadded = FFT.Forward(psfPadded, 1.0);

                return PSFPadded;
            }

            /// <summary>
            /// Correlate the PSF with itself
            /// </summary>
            /// <param name="psfCorrelated"></param>
            /// <returns></returns>
            public static float[,] CalcPSFSquared(Complex[,] psfCorrelated)
            {
                var PSF2 = Fourier2D.Multiply(psfCorrelated, psfCorrelated);
                var psf2 = FFT.Backward(PSF2, psfCorrelated.Length);
                return ToFloatImage(psf2);
            }

            public static void SetPsfInWindow(double[,] psfOutput, double[,] psf, Rectangle window, int yPixel, int xPixel)
            {
                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                    {
                        var y = (yPixel + i) - yPsfHalf;
                        var x = (xPixel + j) - xPsfHalf;
                        if (y >= 0 & y < window.YEnd & x >= 0 & x < window.XEnd)
                        {
                            psfOutput[i + yPsfHalf, j + xPsfHalf] = psf[i, j];
                        }
                        else
                        {
                            psfOutput[i + yPsfHalf, j + yPsfHalf] = 0.0;
                        }
                    }
            }


            /// <summary>
            /// Find maximum pixel value that is outside the cut
            /// </summary>
            /// <param name="fullPsf"></param>
            /// <param name="cutFactor"></param>
            /// <returns></returns>
            public static double CalcMaxSidelobe(double[,] fullPsf, int cutFactor = 2)
            {
                var yOffset = fullPsf.GetLength(0) / 2 - (fullPsf.GetLength(0) / cutFactor) / 2;
                var xOffset = fullPsf.GetLength(1) / 2 - (fullPsf.GetLength(1) / cutFactor) / 2;

                double output = 0.0;
                for (int y = 0; y < fullPsf.GetLength(0); y++)
                    for (int x = 0; x < fullPsf.GetLength(1); x++)
                        if (!(y >= yOffset & y < (yOffset + fullPsf.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + fullPsf.GetLength(1) / cutFactor)))
                            output = Math.Max(output, fullPsf[y, x]);
                return output;
            }
        }

        public static class Residuals
        {
            public static double[,] CalculateBMap(double[,] residuals, Complex[,] psfCorrelation, int yPadding, int xPadding)
            {
                var resPadded = Pad(residuals, yPadding, xPadding);
                var ResPAdded = FFT.Forward(resPadded, 1.0);
                var B = Fourier2D.Multiply(ResPAdded, psfCorrelation);
                var bPadded = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
                var bMap = RemovePadding(bPadded, yPadding, xPadding);
                return bMap;
            }

            public static float[,] CalcBMap(float[,] residuals, Complex[,] psfCorrelation, Rectangle psfSize)
            {
                var yPadding = psfSize.YEnd;
                var xPadding = psfSize.XEnd;

                var resPadded = Pad(residuals, yPadding, xPadding);
                var ResPAdded = FFT.Forward(resPadded, 1.0);
                var B = Fourier2D.Multiply(ResPAdded, psfCorrelation);
                var bPadded = FFT.BackwardFloat(B, (double)(B.GetLength(0) * B.GetLength(1)));
                var bMap = RemovePadding(bPadded, yPadding, xPadding);
                return bMap;
            }

            private static float[,] Pad(float[,] image, int yPadding, int xPadding)
            {
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                var resPadded = new float[image.GetLength(0) + yPadding, image.GetLength(1) + xPadding];
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        resPadded[y + yPsfHalf, x + xPsfHalf] = image[y, x];

                return resPadded;
            }

            private static float[,] RemovePadding(float[,] image, int yPadding, int xPadding)
            {
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                var imgNoPadding = new float[image.GetLength(0) - yPadding, image.GetLength(1) - xPadding];
                for (int y = 0; y < imgNoPadding.GetLength(0); y++)
                    for (int x = 0; x < imgNoPadding.GetLength(1); x++)
                        imgNoPadding[y, x] = image[y + yPsfHalf, x + xPsfHalf];

                return imgNoPadding;
            }

            private static double[,] Pad(double[,] image, int yPadding, int xPadding)
            {
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                var resPadded = new double[image.GetLength(0) + yPadding, image.GetLength(1) + xPadding];
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        resPadded[y + yPsfHalf, x + xPsfHalf] = image[y, x];

                return resPadded;
            }

            private static double[,] RemovePadding(double[,] image, int yPadding, int xPadding)
            {
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                var imgNoPadding = new double[image.GetLength(0) - yPadding, image.GetLength(1) - xPadding];
                for (int y = 0; y < imgNoPadding.GetLength(0); y++)
                    for (int x = 0; x < imgNoPadding.GetLength(1); x++)
                        imgNoPadding[y, x] = image[y + yPsfHalf, x + xPsfHalf];

                return imgNoPadding;
            }
        }

        public static class Fourier2D 
        {
            public static Complex[,] Multiply(Complex[,] vis0, Complex[,] vis1)
            {
                var outputVis = new Complex[vis0.GetLength(0), vis0.GetLength(1)];
                for (int i = 0; i < vis0.GetLength(0); i++)
                    for (int j = 0; j < vis0.GetLength(1); j++)
                        outputVis[i, j] = vis0[i, j] * vis1[i, j];

                return outputVis;
            }

            public static Complex[,] Add(Complex[,] vis0, Complex[,] vis1)
            {
                var outputVis = new Complex[vis0.GetLength(0), vis0.GetLength(1)];
                for (int i = 0; i < vis0.GetLength(0); i++)
                    for (int j = 0; j < vis0.GetLength(1); j++)
                        outputVis[i, j] = vis0[i, j] + vis1[i, j];

                return outputVis;
            }

            public static Complex[,] Subtract(Complex[,] vis0, Complex[,] vis1)
            {
                var outputVis = new Complex[vis0.GetLength(0), vis0.GetLength(1)];
                for (int i = 0; i < vis0.GetLength(0); i++)
                    for (int j = 0; j < vis0.GetLength(1); j++)
                        outputVis[i, j] = vis0[i, j] - vis1[i, j];

                return outputVis;
            }

            public static void SubtractInPlace(Complex[,] vis0, Complex[,] vis1)
            {
                for (int i = 0; i < vis0.GetLength(0); i++)
                    for (int j = 0; j < vis0.GetLength(1); j++)
                        vis0[i, j] -=  vis1[i, j];
            }
        }

        public static class Visibilities
        {
            public static Complex[,,] Substract(Complex[,,] vis0, Complex[,,] vis1, bool[,,] flag)
            {
                var residualVis = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
                for (int i = 0; i < vis0.GetLength(0); i++)
                    for (int j = 0; j < vis0.GetLength(1); j++)
                        for (int k = 0; k < vis0.GetLength(2); k++)
                            if (!flag[i, j, k])
                                residualVis[i, j, k] = vis0[i, j, k] - vis1[i, j, k];
                            else
                                residualVis[i, j, k] = 0;

                return residualVis;
            }
        }

        public static float[,] ToFloatImage(double[,] image)
        {
            var output = new float[image.GetLength(0), image.GetLength(1)];
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); j++)
                    output[i, j] = (float)image[i,j];

            return output;
        }

        public class Rectangle
        {
            public int Y { get; private set; }
            public int X { get; private set; }

            public int YEnd { get; private set; }
            public int XEnd { get; private set; }

            public Rectangle(int y, int x, int yEnd, int xEnd)
            {
                Y = y;
                X = x;
                YEnd = yEnd;
                XEnd = xEnd;
            }

            public int YExtent() => YEnd - Y;

            public int XExtent() => XEnd - X;
        }
    }
}
