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
                        var currentA = QueryIntegral(integral, y, x, res.GetLength(0), res.GetLength(1));//wrong
                        var old = xImage[yLocal, xLocal];
                        var xTmp = old + b[y, x] / currentA;
                        xTmp = GreedyCD.ShrinkPositive(xTmp, lambda * alpha) / (1 + lambda * (1 - alpha));

                        var xDiff = old - xTmp;
                        var oImprov = GreedyCD.EstimateObjectiveImprovement(res, psf, y, x, xDiff);
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
                    
                    
                    
                    //UpdateResiduals(res, resUpdate, psf, yPixel, xPixel, xOld - xNew);
                }
            }

            return true;
        }

        public static double QueryIntegral(double[,] integral, int yPixel, int xPixel, int resYLength, int resXLength)
        {

            var yOverShoot = resYLength * 2 - (yPixel + (integral.GetLength(1) - integral.GetLength(1) / 2) - 1);
            var xOverShoot = resXLength * 2 - (xPixel + (integral.GetLength(1) - integral.GetLength(1) / 2) - 1);

            var yCorrection = yOverShoot % integral.GetLength(0);
            var xCorrection = xOverShoot % integral.GetLength(1);

            if (yCorrection == yOverShoot & xCorrection == xOverShoot)
            {
                return integral[yCorrection, xCorrection];
            }
            else if (yCorrection == yOverShoot | xCorrection == xOverShoot)
            {
                var y = Math.Min(yOverShoot, integral.GetLength(0) - 1);
                var x = Math.Min(xOverShoot, integral.GetLength(1) - 1);
                return integral[y, x] - integral[yCorrection, xCorrection];
            }
            else
            {
                return integral[integral.GetLength(0) - 1, integral.GetLength(1) - 1]
                       - integral[integral.GetLength(0) - 1, xCorrection]
                       - integral[yCorrection, integral.GetLength(1) - 1]
                       + integral[yCorrection, xCorrection];
            }
        }
    }
}
