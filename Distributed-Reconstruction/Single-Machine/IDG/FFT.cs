﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FFTW.NET;
using System.Numerics;

namespace Single_Machine.IDG
{
    class FFT
    {
        public static List<List<Complex[,]>> SubgridFFT(GriddingConstants c, List<List<Complex[,]>> subgrids)
        {
            var output = new List<List<Complex[,]>>(subgrids.Count);
            for (int baseline= 0; baseline < subgrids.Count; baseline++)
            {
                var blSubgrids = subgrids[baseline];
                var blOutput = new List<Complex[,]>(blSubgrids.Count);
                for (int subgrid = 0; subgrid < blSubgrids.Count; subgrid++)
                {
                    var sub = blSubgrids[subgrid];
                    var outFourier = new Complex[c.SubgridSize, c.SubgridSize];
                    using (var imageSpace = new AlignedArrayComplex(16, c.SubgridSize, c.SubgridSize))
                    using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
                    {
                        //copy
                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                imageSpace[i, j] = sub[i, j];
                        }
                        
                        DFT.FFT(imageSpace, fourierSpace);
                        var norm = 1.0 / (c.SubgridSize * c.SubgridSize);

                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                outFourier[i, j] = fourierSpace[i, j] * norm;
                        }
                    }
                    blOutput.Add(outFourier);
                }
                output.Add(blOutput);
            }

            return output;
        }

        public static List<List<Complex[,]>> SubgridIFFT(GriddingConstants c, List<List<Complex[,]>> subgrids)
        {
            var output = new List<List<Complex[,]>>(subgrids.Count);
            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blSubgrids = subgrids[baseline];
                var blOutput = new List<Complex[,]>(blSubgrids.Count);
                for (int subgrid = 0; subgrid < blSubgrids.Count; subgrid++)
                {
                    var sub = blSubgrids[subgrid];
                    var outFourier = new Complex[c.SubgridSize, c.SubgridSize];
                    using (var imageSpace = new AlignedArrayComplex(16, c.SubgridSize, c.SubgridSize))
                    using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
                    {
                        //copy
                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                imageSpace[i, j] = sub[i, j];
                        }

                        DFT.FFT(imageSpace, fourierSpace);
                        var norm = 1.0 / (c.SubgridSize * c.SubgridSize);

                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                outFourier[i, j] = fourierSpace[i, j] * norm;
                        }
                    }
                    blOutput.Add(outFourier);
                }
                output.Add(blOutput);
            }

            return output;
        }

        public static double[,] GridIFFT(Complex[,] grid, long visibilitiesCount = 1)
        {
            double[,] output = new double[grid.GetLength(0), grid.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, grid.GetLength(0), grid.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < grid.GetLength(0); y++)
                {
                    for (int x = 0; x < grid.GetLength(1); x++)
                    {
                        fourierSpace[y, x] = grid[y, x];
                    }
                }

                DFT.IFFT(fourierSpace, imageSpace);

                for (int y = 0; y < grid.GetLength(0); y++)
                {
                    for (int x = 0; x < grid.GetLength(1); x++)
                    {
                        output[y, x] = imageSpace[y, x].Real/visibilitiesCount;
                    }
                }

            }

            return output;
        }


        public static Complex[,] GridFFT(double[,] image)
        {
            Complex[,] output = new Complex[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.FFT(imageSpace, fourierSpace);
                //double norm = 1.0 / (image.GetLength(0) * image.GetLength(1));
                var norm = 1.0;

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x] * norm;
            }

            return output;
        }

        /// <summary>
        /// Just here for debug purposes
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static Complex[,] GridFFT(Complex[,] image)
        {
            Complex[,] output = new Complex[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.FFT(imageSpace, fourierSpace);
                //double norm = 1.0 / (image.GetLength(0) * image.GetLength(1));
                var norm = 1.0;

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x] * norm;
            }

            return output;
        }

        public static void Shift(Complex[,] grid)
        {
            // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
            var n2 = grid.GetLength(0) / 2;
            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    var tmp13 = grid[i, j];
                    grid[i, j] = grid[i + n2, j + n2];
                    grid[i + n2, j + n2] = tmp13;

                    var tmp24 = grid[i + n2, j];
                    grid[i + n2, j] = grid[i, j + n2];
                    grid[i, j + n2] = tmp24;

                }
            }
        }

        public static void Shift(double[,] grid)
        {
            // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
            var n2 = grid.GetLength(0) / 2;
            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    var tmp13 = grid[i, j];
                    grid[i, j] = grid[i + n2, j + n2];
                    grid[i + n2, j + n2] = tmp13;

                    var tmp24 = grid[i + n2, j];
                    grid[i + n2, j] = grid[i, j + n2];
                    grid[i, j + n2] = tmp24;

                }
            }
        }

    }
}