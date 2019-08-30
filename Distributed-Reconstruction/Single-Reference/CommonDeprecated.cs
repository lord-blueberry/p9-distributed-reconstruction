using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace Single_Reference
{
    static class CommonDeprecated
    {
        public static class PSF
        {
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
