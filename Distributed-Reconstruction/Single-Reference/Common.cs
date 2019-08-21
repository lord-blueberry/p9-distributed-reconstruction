using System;
using System.Collections.Generic;
using System.Text;
using Single_Reference.IDGSequential;
using System.Numerics;

namespace Single_Reference
{
    public static class Common
    {
        public static double ShrinkElasticNet(double value, double lambda, double alpha) => Math.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        public static class PSF
        {
            public static double[,] CalcPSFScan(double[,] psf)
            {
                var scan = new double[psf.GetLength(0), psf.GetLength(1)];
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                    {
                        var iBefore = i > 0 ? scan[i - 1, j] : 0.0;
                        var jBefore = j > 0 ? scan[i, j - 1] : 0.0;
                        var ijBefore = i > 0 & j > 0 ? scan[i - 1, j - 1] : 0.0;
                        var current = psf[i, j] * psf[i, j];
                        scan[i, j] = current + iBefore + jBefore - ijBefore;
                    }

                return scan;
            }

            public static double QueryScan(double[,] psfScan, int yPixel, int xPixel, int yLength, int xLength)
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

            public static double[,] CalcAMap(double[,] image, double[,] psf)
            {
                var scan = CalcPSFScan(psf);
                var aMap = new double[image.GetLength(0), image.GetLength(1)];
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        aMap[y, x] = QueryScan(scan, y, x, image.GetLength(0), image.GetLength(1));
                return aMap;
            }

            public static Complex[,] CalculateFourierCorrelation(double[,] psf, int yPadding, int xPadding)
            {
                var psfPadded = new double[yPadding + psf.GetLength(0), xPadding + psf.GetLength(1)];
                var yPsfHalf = yPadding / 2;
                var xPsfHalf = xPadding / 2;
                for (int y = 0; y < psf.GetLength(0); y++)
                    for (int x = 0; x < psf.GetLength(1); x++)
                        psfPadded[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
                FFT.Shift(psfPadded);
                var PSFPadded = FFT.Forward(psfPadded, 1.0);

                return PSFPadded;
            }

            /// <summary>
            /// Calculate PSF*PSF
            /// </summary>
            /// <param name="image">is used to check if parts of the psf are masked by the image dimensions</param>
            /// <param name="psf"></param>
            /// <returns></returns>
            public static double[,] CalcPSFSquared(double[,] image, double[,] psf)
            {
                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;

                //invert the PSF, since we actually do want to correlate the psf with the residuals. (The FFT already inverts the psf, so we need to invert it again to not invert it. Trust me.)
                var psfTmp = new double[psf.GetLength(0) + psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
                for (int y = 0; y < psf.GetLength(0); y++)
                    for (int x = 0; x < psf.GetLength(1); x++)
                        psfTmp[y + yPsfHalf + 1, x + xPsfHalf + 1] = psf[psf.GetLength(0) - y - 1, psf.GetLength(1) - x - 1];
                FFT.Shift(psfTmp);
                var PsfCorr = FFT.Forward(psfTmp, 1.0);

                psfTmp = new double[psf.GetLength(0) + psf.GetLength(0), psf.GetLength(1) + psf.GetLength(1)];
                SetPSFInWindow(psfTmp, image, psf, image.GetLength(0) / 2, image.GetLength(1) / 2);
                var tmp = FFT.Forward(psfTmp, 1.0);
                var tmp2 = Fourier2D.Multiply(tmp, PsfCorr);

                var psf2 = FFT.Backward(tmp2, (double)(tmp2.GetLength(0) * tmp2.GetLength(1)));

                return psf2;
            }

            public static void SetPSFInWindow(double[,] psfPadded, double[,] window, double[,] psf, int yPixel, int xPixel)
            {
                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;
                for (int i = 0; i < psf.GetLength(0); i++)
                    for (int j = 0; j < psf.GetLength(1); j++)
                    {
                        var y = (yPixel + i) - yPsfHalf;
                        var x = (xPixel + j) - xPsfHalf;
                        if (y >= 0 & y < window.GetLength(0) & x >= 0 & x < window.GetLength(1))
                        {
                            psfPadded[i + yPsfHalf, j + xPsfHalf] = psf[i, j];
                        }
                        else
                        {
                            psfPadded[i + yPsfHalf, j + yPsfHalf] = 0.0;
                        }
                    }
            }

            /// <summary>
            /// Find maximum pixel value that is outside the cut
            /// </summary>
            /// <param name="fullPsf"></param>
            /// <param name="cutFactor"></param>
            /// <returns></returns>
            public static double CalculateMaxSidelobe(double[,] fullPsf, int cutFactor = 2)
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
            public static double[,] CalculateBMap(double[,] residuals, Complex[,] psfCorrelated, int yPadding, int xPadding)
            {
                var resPadded = Residuals.Pad(residuals, yPadding, xPadding);
                var ResPAdded = FFT.Forward(resPadded, 1.0);
                var B = Fourier2D.Multiply(ResPAdded, psfCorrelated);
                var bPadded = FFT.Backward(B, (double)(B.GetLength(0) * B.GetLength(1)));
                var bMap = Residuals.RemovePadding(bPadded, yPadding, xPadding);
                return bMap;
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

            public static double[,] Pad(double[,] image, double[,] psf)
            {
                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;
                var resPadded = new double[image.GetLength(0) + psf.GetLength(0), image.GetLength(1) + psf.GetLength(1)];
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        resPadded[y + yPsfHalf, x + xPsfHalf] = image[y, x];

                return resPadded;
            }

            public static double[,] RemovePadding(double[,] img, double[,] psf)
            {
                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;
                var imgNoPadding = new double[img.GetLength(0) - psf.GetLength(0), img.GetLength(1) - psf.GetLength(1)];
                for (int y = 0; y < imgNoPadding.GetLength(0); y++)
                    for (int x = 0; x < imgNoPadding.GetLength(1); x++)
                        imgNoPadding[y, x] = img[y + yPsfHalf, x + xPsfHalf];

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
        }
    }
}
