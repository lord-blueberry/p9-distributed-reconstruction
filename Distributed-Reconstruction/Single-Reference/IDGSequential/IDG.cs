using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Single_Reference.IDGSequential
{
    public class IDG
    {
        public static Complex[,] Grid(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);

            return grid;
        }

        public static Complex[,] GridPSF(GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            var visibilities = new Complex[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            for (int i = 0; i < visibilities.GetLength(0); i++)
                for (int j = 0; j < visibilities.GetLength(1); j++)
                    for (int k = 0; k < visibilities.GetLength(2); k++)
                    {
                        if (!flags[i, j, k])
                            visibilities[i, j, k] = new Complex(1.0, 0);
                        else
                            visibilities[i, j, k] = new Complex(0, 0);
                    }


            return Grid(c, metadata, visibilities, uvw, frequencies);
        }

        public static Complex[,,] DeGrid(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,] grid, double[,,] uvw, double[] frequencies)
        {
            FFT.Shift(grid);
            var ftGridded = Adder.SplitHack(c, metadata, grid);
            var gridded = FFT.SubgridIFFT(c, ftGridded);
            var visibilities = Gridder.BackwardsHack(c, metadata, gridded, uvw, frequencies, c.SubgridSpheroidal);

            return visibilities;
        }

        public static double[,] ToImage(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.GridIFFT(grid, c.VisibilitiesCount);
            FFT.Shift(img);

            //remove spheroidal from grid
            for (int y = 0; y < img.GetLength(0); y++)
                for (int x = 0; x < img.GetLength(1); x++)
                    img[y, x] = img[y, x] / c.GridSpheroidal[y, x];

            return img;
        }

        public static double[,] CalculatePSF(GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
        {
            var visibilities = new Complex[uvw.GetLength(0), uvw.GetLength(1), frequencies.Length];
            for (int i = 0; i < visibilities.GetLength(0); i++)
                for (int j = 0; j < visibilities.GetLength(1); j++)
                    for (int k = 0; k < visibilities.GetLength(2); k++)
                    {
                        if (!flags[i, j, k])
                        {
                            visibilities[i, j, k] = new Complex(1.0, 0);
                        }
                        else
                            visibilities[i, j, k] = new Complex(0, 0);
                    }
                        
            
            var gridded = Gridder.ForwardHack(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = FFT.SubgridFFT(c, gridded);
            var grid = Adder.AddHack(c, metadata, ftgridded);
            FFT.Shift(grid);
            var psf = FFT.GridIFFT(grid, c.VisibilitiesCount);
            FFT.Shift(psf);

            //remove spheroidal from grid
            for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psf[y, x] = psf[y, x] / c.GridSpheroidal[y, x];

            return psf;
        }

        public static Complex[,,] ToVisibilities(GriddingConstants c, List<List<SubgridHack>> metadata, double[,] image, double[,,] uvw, double[] frequencies)
        {
            //add spheroidal to grid?
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); j++)
                    image[i, j] = image[i, j] / c.GridSpheroidal[i, j];

            FFT.Shift(image);
            var grid = FFT.GridFFT(image);
            FFT.Shift(image);

            FFT.Shift(grid);
            var ftGridded = Adder.SplitHack(c, metadata, grid);
            var gridded = FFT.SubgridIFFT(c, ftGridded);
            var visibilities = Gridder.BackwardsHack(c, metadata, gridded, uvw, frequencies, c.SubgridSpheroidal);

            return visibilities;
        }

        public static Complex[,,] Substract(Complex[,,] vis0, Complex[,,] vis1, bool[,,] flag)
        {
            var residualVis = new Complex[vis0.GetLength(0), vis0.GetLength(1), vis0.GetLength(2)];
            for (int i = 0; i < vis0.GetLength(0); i++)
                for (int j = 0; j < vis0.GetLength(1); j++)
                    for (int k = 0; k < vis0.GetLength(2); k++)
                        if (!flag[i, j, k])
                            residualVis[i, j, k] = vis0[i, j, k] - vis1[i, j, k];
                        else
                            residualVis[i, j, k] = 0;

            return residualVis;
        }

        public static Complex[,] Multiply(Complex[,] visGrid0, Complex[,] visGrid1)
        {
            var outputVis = new Complex[visGrid0.GetLength(0), visGrid0.GetLength(1)];
            for (int i = 0; i < visGrid0.GetLength(0); i++)
                for (int j = 0; j < visGrid0.GetLength(1); j++)
                            outputVis[i, j] = visGrid0[i, j] * visGrid1[i, j];

            return outputVis;
        }
    }
}
