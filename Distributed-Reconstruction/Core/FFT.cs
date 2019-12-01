using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FFTW.NET;
using System.Numerics;

namespace Core
{
    public class FFT : IDisposable
    {
        private readonly FftwPlanC2C fft;
        private readonly FftwPlanC2C ifft;
        public AlignedArrayComplex ImageBuffer {get; private set;}
        public AlignedArrayComplex FourierBuffer { get; private set; }

        public FFT(int ySize, int xSize)
            :this(ySize, xSize, Environment.ProcessorCount)
        {
            
        }

        public FFT(int ySize, int xSize, int nCores)
        {
            var dims = new int[] { ySize, xSize };
            ImageBuffer = new AlignedArrayComplex(16, dims);
            FourierBuffer = new AlignedArrayComplex(16, dims);
            fft = FftwPlanC2C.Create(ImageBuffer, FourierBuffer, DftDirection.Forwards, PlannerFlags.Default, nCores);
            ifft = FftwPlanC2C.Create(FourierBuffer, ImageBuffer, DftDirection.Backwards, PlannerFlags.Default, nCores);
        }

        /// <summary>
        /// overwrites FourierBuffer
        /// </summary>
        public void Forward()
        {
            fft.Execute();
        }

        /// <summary>
        /// overwrites ImageBuffer
        /// </summary>
        public void Backward()
        {
            ifft.Execute();
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    fft.Dispose();
                    ifft.Dispose();
                    ImageBuffer.Dispose();
                    FourierBuffer.Dispose();
                }

                ImageBuffer = null;
                FourierBuffer = null;

                disposedValue = true;
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            Dispose(true);
        }
        #endregion

        #region static methods
        public static Complex[,] Forward(double[,] image, double norm = 1.0)
        {
            Complex[,] output = new Complex[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.FFT(imageSpace, fourierSpace, PlannerFlags.Default, Environment.ProcessorCount);

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x] / norm;
            }

            return output;
        }

        public static Complex[,] Forward(float[,] image, double norm = 1.0)
        {
            Complex[,] output = new Complex[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.FFT(imageSpace, fourierSpace);

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x] / norm;
            }

            return output;
        }

        public static double[,] Backward(Complex[,] image, double norm)
        {
            double[,] output = new double[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.IFFT(imageSpace, fourierSpace, PlannerFlags.Default, Environment.ProcessorCount);

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x].Real / norm;
            }

            return output;
        }

        public static float[,] BackwardFloat(Complex[,] image, double norm)
        {
            var output = new float[image.GetLength(0), image.GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, image.GetLength(0), image.GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        imageSpace[y, x] = image[y, x];

                DFT.IFFT(imageSpace, fourierSpace, PlannerFlags.Default, Environment.ProcessorCount);

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = (float)(fourierSpace[y, x].Real / norm);
            }

            return output;
        }

        public static double[,] Backward(Complex[,] grid, long visibilityCount)
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

                DFT.IFFT(fourierSpace, imageSpace, PlannerFlags.Default, Environment.ProcessorCount);

                for (int y = 0; y < grid.GetLength(0); y++)
                {
                    for (int x = 0; x < grid.GetLength(1); x++)
                    {
                        output[y, x] = imageSpace[y, x].Real / visibilityCount;
                    }
                }
            }

            return output;
        }

        #region w-stacking methods
        public static float[,] WStackIFFTFloat(List<Complex[,]> grid, long visibilityCount)
        {
            var output = new float[grid[0].GetLength(0), grid[0].GetLength(1)];
            using (var imageSpace = new AlignedArrayComplex(16, grid[0].GetLength(0), grid[0].GetLength(1)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int k = 0; k < grid.Count; k++)
                {
                    Parallel.For(0, grid[0].GetLength(0), (y) =>
                    {
                        for (int x = 0; x < grid[0].GetLength(1); x++)
                            fourierSpace[y, x] = grid[k][y, x];

                    });

                    DFT.IFFT(fourierSpace, imageSpace, PlannerFlags.Default, Environment.ProcessorCount);

                    Parallel.For(0, grid[0].GetLength(0), (y) =>
                    {
                        for (int x = 0; x < grid[0].GetLength(1); x++)
                            output[y, x] += (float)(imageSpace[y, x].Real / visibilityCount);
                        
                    });
                }
            }

            return output;
        }

        /// <summary>
        /// W-stacking shift
        /// </summary>
        /// <param name="grid"></param>
        public static void Shift(List<Complex[,]> grid)
        {
            Parallel.For(0, grid.Count, (k) =>
            {
                // Interchange entries in 4 quadrants, 1 <-->  and 2 <--> 4
                var n2 = grid[0].GetLength(0) / 2;
                for (int i = 0; i < n2; i++)
                {
                    for (int j = 0; j < n2; j++)
                    {
                        var tmp13 = grid[k][i, j];
                        grid[k][i, j] = grid[k][i + n2, j + n2];
                        grid[k][i + n2, j + n2] = tmp13;

                        var tmp24 = grid[k][i + n2, j];
                        grid[k][i + n2, j] = grid[k][i, j + n2];
                        grid[k][i, j + n2] = tmp24;
                    }
                }
            });
        }
        #endregion

        /// <summary>
        /// Swap the 4 quadrants, 1 <--> 3 and 2 <--> 4
        /// </summary>
        /// <param name="grid"></param>
        public static void Shift<T>(T[,] grid)
        {
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

        public static void Shift(AlignedArrayComplex grid)
        {
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
        #endregion
    }
}
