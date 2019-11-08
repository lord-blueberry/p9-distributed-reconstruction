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
    }
}
