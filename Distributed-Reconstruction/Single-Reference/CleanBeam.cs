using System;
using System.Collections.Generic;
using System.Text;

using System.Threading.Tasks;

namespace Single_Reference
{
    public class CleanBeam
    {
        public static double[,] ConvolveCleanBeam(double[,] image)
        {
            var cleanbeam = FitsIO.ReadBeam("cleanbeam.fits");

            var sumBeam = 0.0;
            for(int i= 0; i < cleanbeam.GetLength(0); i++)
                for(int j = 0; j < cleanbeam.GetLength(1); j++)
                    sumBeam += cleanbeam[i, j];

            var output = new double[image.GetLength(0), image.GetLength(1)];
            Parallel.For(0, image.GetLength(0), i =>
            {
                for(int j = 0; j < image.GetLength(1); j++)
                {
                    var sum = 0.0;
                    for (int k = 0; k < cleanbeam.GetLength(0); k++)
                    {
                        for (int l = 0; l < cleanbeam.GetLength(1); l++)
                        {
                            var y = i + k - cleanbeam.GetLength(0) / 2;
                            var x = j + l - cleanbeam.GetLength(1) / 2;
                            if (y >= 0 & y < image.GetLength(0) &
                                x >= 0 & x < image.GetLength(1))
                            {
                                sum += (image[y, x] * cleanbeam[cleanbeam.GetLength(0) - 1 - k, cleanbeam.GetLength(1) - 1 - l]) / sumBeam;
                            }
                        }
                    }
                    output[i, j] = sum;
                }
                /*Parallel.For(0, image.GetLength(1), j =>
                {
;
                });*/

            });

            return output;
        }
    }
}