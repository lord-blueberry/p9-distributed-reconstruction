using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Core.ImageDomainGridder
{
    public class IDG
    {
        public static Complex[,] Grid(GriddingConstants c, List<List<Subgrid>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            var gridded = Gridder.Forward(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = SubgridFFT.Forward(c, gridded);
            var grid = Adder.Add(c, metadata, ftgridded);
            FFT.Shift(grid);

            return grid;
        }

        public static Complex[,,] DeGrid(GriddingConstants c, List<List<Subgrid>> metadata, Complex[,] grid, double[,,] uvw, double[] frequencies)
        {
            FFT.Shift(grid);
            var ftGridded = Adder.Split(c, metadata, grid);
            var gridded = SubgridFFT.Backward(c, ftGridded);
            var visibilities = Gridder.Backwards(c, metadata, gridded, uvw, frequencies, c.SubgridSpheroidal);

            return visibilities;
        }

        public static List<Complex[,]> GridW(GriddingConstants c, List<List<Subgrid>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            var gridded = Gridder.Forward(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = SubgridFFT.Forward(c, gridded);
            var grid = AdderWStack.Add(c, metadata, ftgridded);
            FFT.Shift(grid);

            return grid;
        }

        public static Complex[,,] DeGridW(GriddingConstants c, List<List<Subgrid>> metadata, Complex[,] grid, double[,,] uvw, double[] frequencies)
        {
            FFT.Shift(grid);
            var ftGridded = AdderWStack.Split(c, metadata, grid);
            var gridded = SubgridFFT.Backward(c, ftGridded);
            var visibilities = Gridder.Backwards(c, metadata, gridded, uvw, frequencies, c.SubgridSpheroidal);

            return visibilities;
        }

        public static Complex[,] GridPSF(GriddingConstants c, List<List<Subgrid>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
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

        public static double[,] ToImage(GriddingConstants c, List<List<Subgrid>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            var gridded = Gridder.Forward(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = SubgridFFT.Forward(c, gridded);
            var grid = Adder.Add(c, metadata, ftgridded);
            FFT.Shift(grid);
            var img = FFT.Backward(grid, c.VisibilitiesCount);
            FFT.Shift(img);

            //remove spheroidal from grid
            /*for (int y = 0; y < img.GetLength(0); y++)
                for (int x = 0; x < img.GetLength(1); x++)
                    img[y, x] = img[y, x] / c.GridSpheroidal[y, x];*/

            return img;
        }

        public static double[,] CalculatePSF(GriddingConstants c, List<List<Subgrid>> metadata, double[,,] uvw, bool[,,] flags, double[] frequencies)
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
                        
            
            var gridded = Gridder.Forward(c, metadata, uvw, visibilities, frequencies, c.SubgridSpheroidal);
            var ftgridded = SubgridFFT.Forward(c, gridded);
            var grid = Adder.Add(c, metadata, ftgridded);
            FFT.Shift(grid);
            var psf = FFT.Backward(grid, c.VisibilitiesCount);
            FFT.Shift(psf);

            //remove spheroidal from grid
            /*for (int y = 0; y < psf.GetLength(0); y++)
                for (int x = 0; x < psf.GetLength(1); x++)
                    psf[y, x] = psf[y, x] / c.GridSpheroidal[y, x];*/

            return psf;
        }
    }
}
