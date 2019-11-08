using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

using static Single_Reference.Common;

namespace Single_Reference.Deconvolution.Approx
{
    public class Approx
    {
        private static double CalcESO(int processorCount, int degreeOfSep, int blockCount) => 1.0 + (degreeOfSep - 1.0) * (processorCount - 1.0) / (Math.Max(1.0, (blockCount - 1)));

        const float ACTIVE_SET_CUTOFF = 1e-8f;
        int threadCount;
        int blockSize;
        float randomFraction;
        float searchFraction;

        float MaxLipschitz;
        float[,] aMap;
        float[,] psf;
        float[,] psf2;
        Rectangle totalSize;

        public Approx(Rectangle totalSize, float[,] psf, int threadCount, int blockSize, float randomFraction, float searchFraction)
        {
            this.totalSize = totalSize;
            this.psf = psf;
            this.psf2 = PSF.CalcPSFSquared(psf);
            MaxLipschitz = (float)PSF.CalcMaxLipschitz(psf);
            this.aMap = PSF.CalcAMap(psf, totalSize);

            this.threadCount = threadCount;
            this.blockSize = blockSize;
            this.randomFraction = randomFraction;
            this.searchFraction = searchFraction;
        }

        public void ResetAMap(float[,] psf)
        {
            var psf2Local = PSF.CalcPSFSquared(psf);
            var maxFull = Residuals.GetMax(psf2Local);
            MaxLipschitz = maxFull;
            aMap = PSF.CalcAMap(psf, totalSize);
            this.psf = psf;

            var maxCut = Residuals.GetMax(psf2);
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                    psf2[i, j] *= (maxFull / maxCut);
        }

        public static List<Tuple<int, int>> GetActiveSet(float[,] xExplore, float[,] gExplore, float lambda, float alpha, float[,] lipschitzMap)
        {
            var debug = new float[xExplore.GetLength(0), xExplore.GetLength(1)];
            var output = new List<Tuple<int, int>>();
            for (int y = 0; y < xExplore.GetLength(0); y++)
                for (int x = 0; x < xExplore.GetLength(1); x++)
                {
                    var lipschitz = lipschitzMap[y, x];
                    var tmp = gExplore[y, x] + xExplore[y, x] * lipschitz;
                    tmp = ElasticNet.ProximalOperator(tmp, lipschitz, lambda, alpha);
                    if (0.0f < Math.Abs(tmp - xExplore[y, x]))
                    {
                        output.Add(new Tuple<int, int>(y, x));
                        debug[y, x] = 1.0f;
                    }
                }
            FitsIO.Write(debug, "activeSet.fits");
            return output;
        }

        private static float GetAbsMax(float[,] xImage, float[,] gradients, float[,] lipschitz, float lambda, float alpha)
        {
            var maxPixels = new float[xImage.GetLength(0)];
            Parallel.For(0, xImage.GetLength(0), (y) =>
            {
                var yLocal = y;

                var currentMax = 0.0f;
                for (int x = 0; x < xImage.GetLength(1); x++)
                {
                    var xLocal = x;
                    var L = lipschitz[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    //var xTmp = old + bMap[y, x] / currentA;
                    //xTmp = ShrinkElasticNet(xTmp, lambda, alpha);
                    var xTmp = ElasticNet.ProximalOperator(old * L + gradients[y, x], L, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax < Math.Abs(xDiff))
                        currentMax = Math.Abs(xDiff);
                }
                maxPixels[yLocal] = currentMax;
            });

            var maxPixel = 0.0f;
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel < maxPixels[i])
                    maxPixel = maxPixels[i];

            return maxPixel;
        }

    }
}
