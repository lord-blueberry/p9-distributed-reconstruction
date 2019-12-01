using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.IO;

using nom.tam.fits;
using nom.tam.util;


namespace Core
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

        public static void Write(double[,,] img, string file = "Outputfile")
        {
            
            for(int k = 0; k < img.GetLength(0); k++)
            {
                var img2 = new double[img.GetLength(1)][];
                for (int i = 0; i < img2.Length; i++)
                {
                    var gg2 = new double[img.GetLength(2)];
                    img2[i] = gg2;
                    for (int j = 0; j < img.GetLength(2); j++)
                    {
                        gg2[j] = img[k, i, j];
                    }
                }

                var f = new Fits();
                var hhdu = FitsFactory.HDUFactory(img2);
                f.AddHDU(hhdu);

                using (BufferedDataStream fstream = new BufferedDataStream(new FileStream(file+k+".fits", FileMode.Create)))
                {
                    f.Write(fstream);
                }
            }
        }

        public static void Write<T>(T[,] img, string file = "Outputfile.fits")
        {
            var img2 = new T[img.GetLength(0)][];
            for (int i = 0; i < img2.Length; i++)
            {
                var row = new T[img.GetLength(1)];
                img2[i] = row;
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    row[j] = img[i, j];
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
        public static float[,] ReadCASAFits(string file)
        {
            Fits f = new Fits(file);
            ImageHDU h = (ImageHDU)f.ReadHDU();
            var imgRaw = (Array[])h.Kernel;
            imgRaw = (Array[])imgRaw[0];
            imgRaw = (Array[])imgRaw[0];

            var output = new float[imgRaw.Length, imgRaw[0].Length];
            for (int i = 0; i < imgRaw.Length; i++)
            {
                var row = (float[])imgRaw[i];
                for (int j = 0; j < row.Length; j++)
                    output[i, j] = row[j];
            }
            return output;
        }

        public static float[,] ReadImage(string file)
        {
            Fits f = new Fits(file);
            ImageHDU h = (ImageHDU)f.ReadHDU();
            var imgRaw = (Array[])h.Kernel;

            var output = new float[imgRaw.Length, imgRaw[0].Length];
            for(int i = 0; i < imgRaw.Length;i++)
            {
                var row = (float[])imgRaw[i];
                for (int j = 0; j < row.Length; j++)
                    output[i, j] = row[j];
            }
            return output;
        }

        public static double[,] ReadImageDouble(string file)
        {
            Fits f = new Fits(file);
            ImageHDU h = (ImageHDU)f.ReadHDU();
            var imgRaw = (Array[])h.Kernel;

            var output = new double[imgRaw.Length, imgRaw[0].Length];
            for (int i = 0; i < imgRaw.Length; i++)
            {
                var row = (double[])imgRaw[i];
                for (int j = 0; j < row.Length; j++)
                    output[i, j] = row[j];
            }
            return output;
        }

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

        public static double[,,] ReadUVW(string file, int fromBl, int toBl)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();

            // Double Cube Dimensions: baseline, time, uvw
            var uvw_raw = (Array[])h.Kernel;
            var time_samples = uvw_raw[0].Length;
            var length = toBl - fromBl;

            var uvw = new double[length, time_samples, 3];
            for (long i = fromBl; i < toBl; i++)
            {
                Array[] bl = (Array[])uvw_raw[i];
                for (int j = 0; j < time_samples; j++)
                {
                    double[] values = (double[])bl[j];
                    uvw[i - fromBl, j, 0] = values[0]; //u
                    uvw[i - fromBl, j, 1] = -values[1]; //v
                    uvw[i - fromBl, j, 2] = values[2]; //w
                }
            }
            return uvw;
        }


        public static int CountBaselines(string file)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();

            // Double Cube Dimensions: baseline, time, uvw
            var uvw_raw = (Array[])h.Kernel;
            var baselines = uvw_raw.Length;

            return baselines;
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

        public static Complex[,,] ReadVisibilities(string file, int fromBl, int toBl, int timessamplesCount, int channelsCount, double norm)
        {
            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();
            //Double cube Dimensions: baseline, time, channel, pol, complex_component
            var vis_raw = (Array[])h.Kernel;
            var length = toBl - fromBl;

            var visibilities = new Complex[length, timessamplesCount, channelsCount];
            for (int i = fromBl; i < toBl; i++)
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
                        visibilities[i - fromBl, j, k] = new Complex(
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

        public static bool[,,] ReadFlags(string file, int fromBl, int toBl, int timessamplesCount, int channelsCount)
        {
            var length = toBl - fromBl;
            var output = new bool[length, timessamplesCount, channelsCount];

            var f = new Fits(file);
            var h = (ImageHDU)f.ReadHDU();
            var flags_raw = (Array[])h.Kernel;
            for (int i = fromBl; i < toBl; i++)
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
                            output[i - fromBl, j, k] = false;
                        else
                            output[i - fromBl, j, k] = true;
                    }
                }
            }

            return output;
        }
        #endregion

        #region stitching
        public static T[,,] Stitch<T>(T[,,] x, T[,,] y)
        {
            var output = new T[x.GetLength(0) + y.GetLength(0), x.GetLength(1), x.GetLength(2)];
            for (int i = 0; i < x.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    for (int k = 0; k < x.GetLength(2); k++)
                        output[i, j, k] = x[i, j, k];

            for (int i = 0; i < y.GetLength(0); i++)
                for (int j = 0; j < x.GetLength(1); j++)
                    for (int k = 0; k < x.GetLength(2); k++)
                        output[i + x.GetLength(0), j, k] = y[i, j, k];

            return output;
        }
        #endregion
    }
}
