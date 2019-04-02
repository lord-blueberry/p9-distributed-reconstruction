using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Single_Machine.Deconvolution;

namespace Single_Machine_Test
{
    [TestClass]
    public class CDTest
    {
        [TestMethod]
        public void TestSynthetic0()
        {
            var imSize = 64;
            var psfSize = 4;
            var image = new double[imSize, imSize];
            var psf = new double[psfSize, psfSize];

            var psfSum = 29.0;
            psf[1, 1] = 1 / psfSum;
            psf[1, 2] = 2 / psfSum;
            psf[1, 3] = 3 / psfSum;
            psf[2, 1] = 3 / psfSum;
            psf[2, 2] = 8 / psfSum;
            psf[2, 3] = 2 / psfSum;
            psf[3, 1] = 5 / psfSum;
            psf[3, 2] = 3 / psfSum;
            psf[3, 3] = 2 / psfSum;

            var imgOffset = 38;
            var peak = 15.0;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    image[imgOffset + i, imgOffset + j] = peak * psf[1 + i, 1 + j];
                }
            }

            var xImage = new double[imSize, imSize];
            CDClean.CoordinateDescent(xImage, image, psf, 10.0);

            var precision = 1e-6;
            for(int i = 0; i < image.GetLength(0); i++)
            {
                for(int j = 0; j < image.GetLength(1);j++)
                {
                    if (i == 39 & j == 39)
                        Assert.AreEqual(5.0, xImage[i, j], precision);
                    else
                        Assert.AreEqual(0.0, xImage[i, j], precision);
                }
            }

            Assert.AreEqual(10.0 * psf[2, 2], image[39, 39], precision);
        }
    }
}
