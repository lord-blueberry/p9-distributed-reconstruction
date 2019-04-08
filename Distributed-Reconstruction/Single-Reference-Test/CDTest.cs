using Microsoft.VisualStudio.TestTools.UnitTesting;
using Single_Reference.Deconvolution;

namespace Single_Reference_Test
{
    [TestClass]
    public class CDTest
    {
        private static double[,] Convolve(double[,] image, double[,] kernel)
        {
            var output = new double[image.GetLength(0), image.GetLength(1)];
            for (int y = 0; y < image.GetLength(0); y++)
            {
                for (int x = 0; x < image.GetLength(1); x++)
                {
                    double sum = 0;
                    for (int yk = 0; yk < kernel.GetLength(0); yk++)
                    {
                        for (int xk = 0; xk < kernel.GetLength(1); xk++)
                        {
                            int ySrc = y + yk - ((kernel.GetLength(0) - 1) - kernel.GetLength(0) / 2);
                            int xSrc = x + xk - ((kernel.GetLength(1) - 1) - kernel.GetLength(1) / 2);
                            if (ySrc >= 0 & ySrc < image.GetLength(0) &
                                xSrc >= 0 & xSrc < image.GetLength(1))
                            {
                                sum += image[ySrc, xSrc] * kernel[kernel.GetLength(0) - 1 - yk, kernel.GetLength(1) - 1 - xk];
                            }
                        }
                    }
                    output[y, x] = sum;
                }
            }
            return output;
        }

        [TestMethod]
        public void TestConvergence0()
        {
            var imSize = 64;
            var psfSize = 4;
            var psf = new double[psfSize, psfSize];

            var psfSum = 8.0;
            psf[1, 1] = 1 / psfSum;
            psf[1, 2] = 2 / psfSum;
            psf[1, 3] = 3 / psfSum;
            psf[2, 1] = 3 / psfSum;
            psf[2, 2] = 8 / psfSum;
            psf[2, 3] = 2 / psfSum;
            psf[3, 1] = 5 / psfSum;
            psf[3, 2] = 3 / psfSum;
            psf[3, 3] = 2 / psfSum;

            var groundTruth = new double[imSize, imSize];
            groundTruth[33, 33] = 15.0;

            var image = CDTest.Convolve(groundTruth, psf);
            var reconstruction = new double[imSize, imSize];
            CDClean.CoordinateDescent(reconstruction, image, psf, 0.1);

            var precision = 0.1;
            for (int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    Assert.AreEqual(groundTruth[y, x], reconstruction[y, x], precision);
        }


        [TestMethod]
        public void TestConvergence1()
        {
            var imSize = 64;
            var psfSize = 4;
            var psf = new double[psfSize, psfSize];

            var psfSum = 8.0;
            psf[1, 1] = 1 / psfSum;
            psf[1, 2] = 2 / psfSum;
            psf[1, 3] = 3 / psfSum;
            psf[2, 1] = 3 / psfSum;
            psf[2, 2] = 8 / psfSum;
            psf[2, 3] = 2 / psfSum;
            psf[3, 1] = 5 / psfSum;
            psf[3, 2] = 3 / psfSum;
            psf[3, 3] = 2 / psfSum;

            var groundTruth = new double[imSize, imSize];
            groundTruth[33, 33] = 15.0;
            groundTruth[32, 33] = 3.0;
            groundTruth[31, 33] = 2.0;
            groundTruth[32, 32] = 5.0;
            var convolved = CDTest.Convolve(groundTruth, psf);
            var reconstruction = new double[imSize, imSize];
            CDClean.CoordinateDescent(reconstruction, convolved, psf, 0.1);

            var precision = 0.1;
            for (int y = 0; y < convolved.GetLength(0); y++)
                for (int x = 0; x < convolved.GetLength(1); x++)
                    Assert.AreEqual(groundTruth[y, x], reconstruction[y, x], precision);
        }
    }
}
