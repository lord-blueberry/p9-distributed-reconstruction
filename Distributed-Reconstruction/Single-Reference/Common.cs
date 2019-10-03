using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Single_Reference
{
    public static class Common
    {

        public static float ElasticNetPenalty(float value, float alpha) => 1.0f / 2.0f * (1 - alpha) * (value * value) + alpha * Math.Abs(value);

        public static class ElasticNet
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static float ProximalOperator(float x, float lipschitz, float lambda, float alpha) =>
                Math.Max(x - (lambda * alpha), 0f) / (lipschitz + lambda * (1f - alpha));

            public static double CalcPenalty(float[,] image, float lambda, float alpha)
            {
                double output = 0;
                for (int i = 0; i < image.GetLength(0); i++)
                    for (int j = 0; j < image.GetLength(1); j++)
                    {
                        output += lambda * ElasticNetPenalty(image[i, j], alpha);
                    }

                return output;
            }
        }

        public static class PSF
        {
            private static double QueryScan(double[,] psfScan, int yPixel, int xPixel, int yLength, int xLength)
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

                var correction = 0.0;
                if (yUnderShoot > 0)
                    correction += psfScan[yUnderShootIdx, psfScan.GetLength(1) - xOverShoot - 1];
                if (xUnderShoot > 0)
                    correction += psfScan[psfScan.GetLength(0) - yOverShoot - 1, xUnderShootIdx];

                return psfScan[psfScan.GetLength(0) - 1 - yOverShoot, psfScan.GetLength(1) - 1 - xOverShoot] - correction;
            }
            public static float[,] CalcAMap(float[,] psf, Rectangle totalSize, Rectangle imageSection)
            {
                //scan algorithm. This uses double precision. With more realistic psf sizes, single precision became inaccurate
                var scan = new double[psf.GetLength(0), psf.GetLength(1)];
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                    {
                        var iBefore = i > 0 ? scan[i - 1, j] : 0.0f;
                        var jBefore = j > 0 ? scan[i, j - 1] : 0.0f;
                        var ijBefore = i > 0 & j > 0 ? scan[i - 1, j - 1] : 0.0f;
                        var current = psf[i, j] * psf[i, j];
                        scan[i, j] = current + iBefore + jBefore - ijBefore;
                    }

                var aMap = new float[imageSection.YExtent(), imageSection.XExtent()];
                for (int y = imageSection.Y; y < imageSection.YEnd; y++)
                    for (int x = imageSection.X; x < imageSection.XEnd; x++)
                    {
                        var yLocal = y - imageSection.Y;
                        var xLocal = x - imageSection.X;
                        aMap[yLocal, xLocal] = (float)QueryScan(scan, y, x, totalSize.YEnd, totalSize.XEnd);
                    }

                return aMap;
            }

            public static double CalcMaxLipschitz(float[,] psf)
            {
                var squaredSum = 0.0;
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                        squaredSum += psf[i, j] * psf[i, j];
                return squaredSum;
            }

            /// <summary>
            /// Prepares the CORRELATION kernel in Fourier space. It inverts the PSF and applies the FFT. (multiplication in fourier space == convolution, multiplication with inverted kernel in fourier space == correlation)
            /// its padded, otherwise the FFT would calculate the circular convolution which is physically implausible
            /// </summary>
            /// <param name="psf"></param>
            /// <param name="padding">padding to be used. Use the total image size as padding</param>
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
            /// Prepares the CONVOLUTION kernel in Fourier space. It pads PSF and applies the FFT. (multiplication in fourier space == convolution, multiplication with inverted kernel in fourier space == correlation)
            /// its padded, otherwise the FFT would calculate the circular convolution which is physically implausible
            /// </summary>
            /// <param name="psf"></param>
            /// <param name="padding"></param>
            /// <returns></returns>
            public static Complex[,] CalcPaddedFourierConvolution(float[,] psf, Rectangle padding)
            {
                var psfPadded = Pad(psf, padding.YExtent(), padding.XExtent());
                FFT.Shift(psfPadded);
                var PSFPadded = FFT.Forward(psfPadded, 1.0);

                return PSFPadded;
            }

            /// <summary>
            /// Correlate the PSF with itself, and calculate psf squared
            /// </summary>
            /// <param name="psf"></param>
            /// <returns></returns>
            public static float[,] CalcPSFSquared(float[,] psf)
            {
                var psfCorrelated = CalcPaddedFourierCorrelation(psf, new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1)));
                var psfPadded = new float[psf.GetLength(0) * 2, psf.GetLength(1) * 2];
                var fullWindow = new Rectangle(0, 0, psfPadded.GetLength(0), psfPadded.GetLength(1));
                SetPsfInWindow(psfPadded, psf, fullWindow, psf.GetLength(0), psf.GetLength(1));
                var PSF = FFT.Forward(psfPadded);

                //convolve psf with its flipped version == correlation
                var PSF2 = Fourier2D.Multiply(PSF, psfCorrelated);
                var psf2 = FFT.Backward(PSF2, psfCorrelated.Length);
                return ToFloatImage(psf2);
            }

            private static void SetPsfInWindow(float[,] psfOutput, float[,] psf, Rectangle window, int yPixel, int xPixel)
            {
                ///TODO: is this method still necessary??
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
                            psfOutput[i + yPsfHalf, j + yPsfHalf] = 0.0f;
                        }
                    }
            }

            /// <summary>
            /// Find maximum pixel value that is outside the cut
            /// </summary>
            /// <param name="fullPsf"></param>
            /// <param name="cutFactor"></param>
            /// <returns></returns>
            public static float CalcMaxSidelobe(float[,] fullPsf, int cutFactor = 2)
            {
                var yOffset = fullPsf.GetLength(0) / 2 - (fullPsf.GetLength(0) / cutFactor) / 2;
                var xOffset = fullPsf.GetLength(1) / 2 - (fullPsf.GetLength(1) / cutFactor) / 2;

                var output = 0.0f;
                for (int y = 0; y < fullPsf.GetLength(0); y++)
                    for (int x = 0; x < fullPsf.GetLength(1); x++)
                        if (!(y >= yOffset & y < (yOffset + fullPsf.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + fullPsf.GetLength(1) / cutFactor)))
                            output = Math.Max(output, fullPsf[y, x]);
                return output;
            }

            public static float[,] Cut(float[,] psf, int factor = 2)
            {
                var output = new float[psf.GetLength(0) / factor, psf.GetLength(1) / factor];
                var yOffset = psf.GetLength(0) / 2 - output.GetLength(0) / 2;
                var xOffset = psf.GetLength(1) / 2 - output.GetLength(1) / 2;

                for (int y = 0; y < output.GetLength(0); y++)
                    for (int x = 0; x < output.GetLength(0); x++)
                        output[y, x] = psf[yOffset + y, xOffset + x];
                return output;
            }
        }

        public class Residuals
        {
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

            public static float GetMax(float[,] image)
            {
                var max = 0.0f;
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        max = Math.Max(max, image[y, x]);
                return max;
            }

            public static double CalcPenalty(float[,] residuals)
            {
                double output = 0;
                for (int i = 0; i < residuals.GetLength(0); i++)
                    for (int j = 0; j < residuals.GetLength(1); j++)
                        output += residuals[i, j] * residuals[i, j];
                return 0.5 * output;
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

        public static int CountNonZero(float[,] psf)
        {
            var count = 0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    if (psf[y, x] != 0.0)
                        count++;
            return count;
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

            public bool PointInRectangle(int pointY, int pointX)
            {
                bool output = pointY >= Y  & pointY < YEnd;
                output = output & pointX >= X  & pointX < XEnd;
                return output;
            }
        }
    }
}
