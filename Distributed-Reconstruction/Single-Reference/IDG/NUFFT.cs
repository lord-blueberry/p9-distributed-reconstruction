using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Single_Reference.IDG
{
    class NUFFT
    {
        public static double[,] ToImage(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, visibilitiesCount);
            FFT.Shift(img);

            //remove spheroidal from grid
            for (int y = 0; y < img.GetLength(0); y++)
                for (int x = 0; x < img.GetLength(1); x++)
                    img[y, x] = img[y, x] / c.GridSpheroidal[y, x];

            return img;
        }

        public static double[,] CalculatePSF(GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            var visibilities = new Complex[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            for (int i = 0; i < visibilities.GetLength(0); i++)
                for (int j = 0; j < visibilities.GetLength(1); j++)
                    for (int k = 0; k < visibilities.GetLength(2); k++)
                        visibilities[i, j, k] = new Complex(1.0, 0);
            
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);
            var psf = FFT.GridIFFT(grid, visibilitiesCount);
            FFT.Shift(psf);

            //remove spheroidal from grid
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psf[y, x] = psf[y, x] / c.GridSpheroidal[y, x];

            return psf;
        }

        public static double[,] CalculateNormedPSF(GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            var psf = CalculatePSF(c, metadata, uvw, frequencies, visibilitiesCount);

            var sum = 0.0;
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    sum += psf[y, x];

            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psf[y, x] = psf[y, x] / sum;
            return psf;
        }

        public static Complex[,,] ToVisibilities(GriddingConstants c, List<List<SubgridHack>> metadata, double[,] image, double[,,] uvw, double[] frequencies, long visibilitiesCount)
        {
            //add spheroidal to grid?
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); j++)
                    image[i, j] = image[i, j] / c.GridSpheroidal[i, j];

            FFT.Shift(image);
            var grid = FFT.GridFFT(image, visibilitiesCount);
            FFT.Shift(grid);
            var ftGridded = Adder.SplitHack(c, metadata, grid);
            var gridded = FFT.SubgridIFFT(c, ftGridded);
            var visibilities = Gridder.BackwardsHack(c, metadata, gridded, uvw, frequencies, c.SubgridSpheroidal);

            return visibilities;
        }
    }
}
