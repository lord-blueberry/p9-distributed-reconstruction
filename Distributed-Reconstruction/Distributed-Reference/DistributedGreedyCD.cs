using System;
using System.Collections.Generic;
using System.Text;
using MPI;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;

namespace Distributed_Reference
{
    class DistributedGreedyCD
    {
        public class Rectangle
        {
            public int X { get; private set; }
            public int Y { get; private set; }
            public int XLength { get; private set; }
            public int YLength { get; private set; }

            public Rectangle(int x, int y, int xLen, int yLen)
            {
                X = x;
                Y = y;
                XLength = xLen;
                YLength = yLen;
            }
        }

        public class Candidate
        {
            public double OImprov { get; private set; }
            public double XDiff { get; private set; }

            public int YPixel { get; private set; }
            public int XPixel { get; private set; }

            
            public Candidate(double o, double xDiff, int y, int x)
            {
                OImprov = o;
                XDiff = xDiff;
                YPixel = y;
                XPixel = x;
            }

            public override bool Equals(object obj)
            {
                if (obj == null || this.GetType() != obj.GetType())
                    return base.Equals(obj);

                var c = (Candidate)obj;
                var o = this.OImprov == c.OImprov;
                var xd = XDiff == c.XDiff;
                var y = YPixel == c.YPixel;
                var x = XPixel == c.XPixel;
                return (o & xd & y & x);
            }
        }

        public static bool Deconvolve(Intracommunicator comm, double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var integral = GreedyCD.CalcPSf2Integral(psf); 

            //padded
            var resPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            for (int y = 0; y < res.GetLength(0); y++)
                for (int x = 0; x < res.GetLength(1); x++)
                    resPadded[y + yPsfHalf, x + xPsfHalf] = res[y, x];

            var psfPadded = new double[res.GetLength(0) + psf.GetLength(0), res.GetLength(1) + psf.GetLength(1)];
            var psfYOffset = res.GetLength(0) / 2;
            var psfXOffset = res.GetLength(1) / 2;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psfPadded[y + psfYOffset, x + psfXOffset] = psf[y, x];

            var RES = FFT.FFTDebug(resPadded, 1.0);
            var PSFPadded = FFT.FFTDebug(psfPadded, 1.0);
            var B = IDG.Multiply(RES, PSFPadded);
            var b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
            FFT.Shift(b);

            //Greedy Coordinate Descent
            int iter = 0;
            bool converged = false;
            double epsilon = 1e-4;
            while (!converged & iter < maxIteration)
            {
                var yPixel = -1;
                var xPixel = -1;
                var maxImprov = 0.0;
                var xNew = 0.0;
                for (int y = rec.Y; y < rec.YLength; y++)
                    for (int x = rec.X; x < rec.XLength; x++)
                    {
                        var yLocal = y - rec.Y;
                        var xLocal = x - rec.X;
                        var currentA = QueryIntegral(integral, y, x, res.GetLength(0), res.GetLength(1));
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y + yPsfHalf, x + xPsfHalf] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;
                        var oImprov = EstimateObjectiveImprovement(resPadded, res, psf, y, x, xDiff);
                        var lambdaA = lambda * 2 * currentA;
                        oImprov += lambdaA * GreedyCD.ElasticNetRegularization(old, alpha);
                        oImprov -= lambdaA * GreedyCD.ElasticNetRegularization(xTmp, alpha);

                        if (oImprov > maxImprov)
                        {
                            yPixel = y;
                            xPixel = x;
                            maxImprov = oImprov;
                            xNew = xTmp;
                        }
                    }

                //exchange max
                var xOld = xImage[yPixel, xPixel];
                var candidate = new Candidate(maxImprov, xOld - xNew, yPixel, xPixel);
                var maxCandidate = comm.Allreduce(candidate, (aC, bC) => aC.OImprov > bC.OImprov ? aC : bC);
                converged = maxImprov < epsilon;
                if (!converged)
                {
                    if(maxCandidate == candidate)
                    {
                        var yLocal = yPixel - rec.Y;
                        var xLocal = xPixel - rec.X;
                        xImage[yLocal, xLocal] = xNew;
                    }

                    UpdateResiduals(resPadded, res, psf, yPixel, xPixel, xOld - xNew, yPsfHalf, xPsfHalf);
                    RES = FFT.FFTDebug(resPadded, 1.0);
                    B = IDG.Multiply(RES, PSFPadded);
                    b = FFT.IFFTDebug(B, B.GetLength(0) * B.GetLength(1));
                    FFT.Shift(b);
                }
            }

            return true;
        }

        public static double QueryIntegral(double[,] integral, int yPixel, int xPixel, int resYLength, int resXLength)
        {
            var yOverShoot = (yPixel + (integral.GetLength(0) - integral.GetLength(0) / 2) - 1) - resYLength;
            var xOverShoot = (xPixel + (integral.GetLength(1) - integral.GetLength(1) / 2) - 1) - resXLength;
            var yUnderShoot = (-1) * (yPixel - integral.GetLength(0) / 2);
            var xUnderShoot = (-1) * (xPixel - integral.GetLength(1) / 2);

            //PSF completely in picture
            if (yOverShoot <= 0 & xOverShoot <= 0 & yUnderShoot <= 0 & xUnderShoot <= 0)
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1];

            if(yUnderShoot > 0 & xUnderShoot > 0)
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1]
                       - integral[integral.GetLength(0) - 1, xUnderShoot]
                       - integral[yUnderShoot, integral.GetLength(1) - 1]
                       + integral[yUnderShoot, xUnderShoot];

            yOverShoot = Math.Max(1, yOverShoot) - 1;
            xOverShoot = Math.Max(1, xOverShoot) - 1;
            var correction = 0.0;
            if (yUnderShoot > 0)
                correction = integral[yUnderShoot, integral.GetLength(1) - xOverShoot];
            if (xUnderShoot > 0)
                correction = integral[integral.GetLength(0) - yOverShoot, yUnderShoot];

            return integral[integral.GetLength(0) - yOverShoot, integral.GetLength(1) - xOverShoot] - correction;
        }

        public static void UpdateResiduals(double[,] resPadded, double[,] residuals, double[,] psf, int yPixel, int xPixel, double xDiff, int resYOffset, int resXOffset)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            for (int i = 0; i < psf.GetLength(0); i++)
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < residuals.GetLength(0) & x >= 0 & x < residuals.GetLength(1))
                    {
                        resPadded[y + resYOffset, x + resXOffset] += psf[i, j] * xDiff;
                    }
                }
        }

        public static double EstimateObjectiveImprovement(double[,] resPadded, double[,] res, double[,] psf, int yPixel, int xPixel, double xDiff)
        {
            var yPsfHalf = psf.GetLength(0) / 2;
            var xPsfHalf = psf.GetLength(1) / 2;
            var totalDiff = 0.0;
            var resOld = 0.0;
            for (int i = 0; i < psf.GetLength(0); i++)
            {
                for (int j = 0; j < psf.GetLength(1); j++)
                {
                    var y = (yPixel + i) - yPsfHalf;
                    var x = (xPixel + j) - xPsfHalf;
                    if (y >= 0 & y < res.GetLength(0) &
                        x >= 0 & x < res.GetLength(1))
                    {
                        var resVal = resPadded[y + yPsfHalf, x + xPsfHalf];
                        var newRes = resVal + psf[i, j] * xDiff;
                        resOld += (resVal * resVal);
                        totalDiff += (newRes * newRes);
                    }
                }
            }

            return resOld - totalDiff;
        }
    }
}
