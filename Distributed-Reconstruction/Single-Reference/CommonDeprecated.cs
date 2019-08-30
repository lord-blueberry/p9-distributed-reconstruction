using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace Single_Reference
{
    [Obsolete("Only here so toy examples still work")]
    static class CommonDeprecated
    {
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

            public static Complex[,] CalculateFourierCorrelation(double[,] psf, int yPadding, int xPadding)
            {
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
        }
    }
}
