using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.IO;

using nom.tam.fits;
using nom.tam.util;


namespace Single_Reference
{
    public class FitsIO
    {
        #region writing
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

        public static double[,] ReadImage(string file)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();

            var raw = (Array[])h.Kernel;
            raw = (Array[])raw[0];
            raw = (Array[])raw[0];
            var img = new double[raw.Length, raw.Length];
            for(int i = 0; i < raw.Length; i++)
            {
                var col = (float[])raw[i];
                for(int j = 0; j < col.Length; j++)
                {
                    img[i, j] = col[j];
                }
            }

            return img;
        }


        public static double[,] ReadBeam(string file)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();

            var raw = (Array[])h.Kernel;
            var img = new double[raw.Length, raw.Length];
            for (int i = 0; i < raw.Length; i++)
            {
                var col = (double[])raw[i];
                for (int j = 0; j < col.Length; j++)
                {
                    img[i, j] = col[j];
                }
            }

            return img;
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
        #endregion

        #region reading
        public static double[] ReadFrequencies(string file)
        {
            Fits f = new Fits(file);
            ImageHDU h = (ImageHDU)f.ReadHDU();
            return (double[])h.Kernel;
        }

        /// <summary>
        /// NOTE: Flips v dimension
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public static double[,,] ReadUVW(string file)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();

            // Double Cube Dimensions: baseline, time, uvw
            var uvw_raw = (Array[])h.Kernel;
            var baselines = uvw_raw.Length;
            var time_samples = uvw_raw[0].Length;

            var uvw = new double[baselines, time_samples, 3];
            for (int i = 0; i < baselines; i++)
            {
                Array[] bl = (Array[])uvw_raw[i];
                for (int j = 0; j < time_samples; j++)
                {
                    double[] values = (double[])bl[j];
                    uvw[i, j, 0] = values[0]; //u
                    uvw[i, j, 1] = -values[1]; //v
                    uvw[i, j, 2] = values[2]; //w
                }
            }
            return uvw;
        }

        /// <summary>
        /// Read Intensity Visibilities
        /// </summary>
        /// <param name="file"></param>
        /// <param name="baselinesCount"></param>
        /// <param name="timessamplesCount"></param>
        /// <param name="channelsCount"></param>
        /// <returns></returns>
        public static Complex[,,] ReadVisibilities(string file, int baselinesCount, int timessamplesCount, int channelsCount, double norm)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();
            //Double cube Dimensions: baseline, time, channel, pol, complex_component
            var vis_raw = (Array[])h.Kernel;
            var visibilities = new Complex[baselinesCount, timessamplesCount, channelsCount];
            for (int i = 0; i < baselinesCount; i++)
            {
                Array[] bl = (Array[])vis_raw[i];
                for (int j = 0; j < timessamplesCount; j++)
                {
                    Array[] times = (Array[])bl[j];
                    for (int k = 0; k < channelsCount; k++)
                    {
                        Array[] channel = (Array[])times[k];
                        double[] pol_XX = (double[])channel[0];
                        double[] pol_YY = (double[])channel[3];

                        //add polarizations XX and YY together to form Intensity Visibilities only
                        visibilities[i, j, k] = new Complex(
                            (pol_XX[0] + pol_YY[0]) / norm,
                            (pol_XX[1] + pol_YY[1]) / norm);
                    }
                }
            }
            return visibilities;
        }

        public static bool[,,] ReadFlags(string file, int baselinesCount, int timessamplesCount, int channelsCount)
        {
            var output = new bool[baselinesCount, timessamplesCount, channelsCount];

            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();
            var flags_raw = (Array[])h.Kernel;
            for (int i = 0; i < baselinesCount; i++)
            {
                Array[] bl = (Array[])flags_raw[i];
                for (int j = 0; j < timessamplesCount; j++)
                {
                    Array[] times = (Array[])bl[j];
                    for (int k = 0; k < channelsCount; k++)
                    {
                        double[] pols = (double[])times[k];
                        var sum = 0.0;
                        for (int l = 0; l < pols.Length; l++)
                            sum += pols[l];

                        //visibilitiy has been flagged                        
                        if (sum == 0)
                            output[i, j, k] = false;
                        else
                            output[i, j, k] = true;
                    }
                }
            }

            return output;
        }
        #endregion
    }
}
