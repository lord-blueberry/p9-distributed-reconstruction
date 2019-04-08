using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.IO;

using nom.tam.fits;
using nom.tam.util;


namespace Single_Reference
{
    class FitsIO
    {
        public static void Write(Complex[,] img, string file = "Outputfile.fits")
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j].Real;
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file, FileMode.Create)))
            {
                f.Write(fstream);
            }
        }
        public static void Write(double[,] img, string file = "Outputfile.fits")
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j];
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file, FileMode.Create)))
            {
                f.Write(fstream);
            }
        }

        public static void WriteImag(Complex[,] img, string file = "Outputfile_imag.fits")
        {
            var img2 = new double[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var gg2 = new double[img.GetLength(1)];
                img2[i] = gg2;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    gg2[j] = img[i, j].Imaginary;
                }
            }

            var f = new Fits();
            var hhdu = FitsFactory.HDUFactory(img2);
            f.AddHDU(hhdu);

            using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file, FileMode.Create)))
            {
                f.Write(fstream);
            }
        }
    }
}
