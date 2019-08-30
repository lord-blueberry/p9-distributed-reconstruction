using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FFTW.NET;
using System.Numerics;

namespace Single_Reference
{
    public class FFT : IDisposable
    {
        private FftwPlanRC fft;
        private FftwPlanRC ifft;
        public AlignedArrayDouble ImageBuffer {get; private set;}
        public AlignedArrayComplex FourierBuffer { get; private set; }

        public FFT(int ySize, int xSize)
            :this(ySize, xSize, Environment.ProcessorCount)
        {
            
        }

        public FFT(int ySize, int xSize, int nCores)
        {
            var dims = new int[] { ySize, xSize };
            ImageBuffer = new AlignedArrayDouble(16, dims);
            FourierBuffer = new AlignedArrayComplex(16, dims);
            fft = FftwPlanRC.Create(ImageBuffer, FourierBuffer, DftDirection.Forwards, PlannerFlags.Default, nCores);
            ifft = FftwPlanRC.Create(ImageBuffer, FourierBuffer, DftDirection.Backwards, PlannerFlags.Default, nCores);
        }

        public void CopyToImageBuffer(float[,] image)
        {
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); i++)
                    ImageBuffer[i, j] = image[i, j];
        }

        public Complex[,] CopyFromFourierBuffer()
        {
            var output = new Complex[FourierBuffer.GetLength(0), FourierBuffer.GetLength(1)];
            for (int i = 0; i < output.GetLength(0); i++)
                for (int j = 0; j < output.GetLength(1); i++)
                    output[i, j] = FourierBuffer[i, j];
            return output;
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

                DFT.IFFT(imageSpace, fourierSpace);

                for (int y = 0; y < image.GetLength(0); y++)
                    for (int x = 0; x < image.GetLength(1); x++)
                        output[y, x] = fourierSpace[y, x].Real / norm;
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

                DFT.IFFT(fourierSpace, imageSpace);

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
        public static double[,,] GridIFFT(Complex[,,] grid, long visibilityCount)
        {
            var output = new double[grid.GetLength(0), grid.GetLength(1), grid.GetLength(2)];
            using (var imageSpace = new AlignedArrayComplex(16, grid.GetLength(1), grid.GetLength(2)))
            using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
            {
                for (int k = 0; k < grid.GetLength(0); k++)
                {
                    for (int y = 0; y < grid.GetLength(1); y++)
                    {
                        for (int x = 0; x < grid.GetLength(2); x++)
                        {
                            fourierSpace[y, x] = grid[k, y, x];
                        }
                    }

                    DFT.IFFT(fourierSpace, imageSpace);

                    for (int y = 0; y < grid.GetLength(1); y++)
                    {
                        for (int x = 0; x < grid.GetLength(2); x++)
                        {
                            output[k, y, x] += imageSpace[y, x].Real / visibilityCount;
                        }
                    }
                }
            }

            return output;
        }

        public static void Shift(double[,,] grid)
        {
            for (int k = 0; k < grid.GetLength(0); k++)
            {
                // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
                var n2 = grid.GetLength(1) / 2;
                for (int i = 0; i < n2; i++)
                {
                    for (int j = 0; j < n2; j++)
                    {
                        var tmp13 = grid[k, i, j];
                        grid[k, i, j] = grid[k, i + n2, j + n2];
                        grid[k, i + n2, j + n2] = tmp13;

                        var tmp24 = grid[k, i + n2, j];
                        grid[k, i + n2, j] = grid[k, i, j + n2];
                        grid[k, i, j + n2] = tmp24;
                    }
                }
            }
        }

        public static void Shift(Complex[,,] grid)
        {
            for (int k = 0; k < grid.GetLength(0); k++)
            {
                // Interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
                var n2 = grid.GetLength(1) / 2;
                for (int i = 0; i < n2; i++)
                {
                    for (int j = 0; j < n2; j++)
                    {
                        var tmp13 = grid[k, i, j];
                        grid[k, i, j] = grid[k, i + n2, j + n2];
                        grid[k, i + n2, j + n2] = tmp13;

                        var tmp24 = grid[k, i + n2, j];
                        grid[k, i + n2, j] = grid[k, i, j + n2];
                        grid[k, i, j + n2] = tmp24;
                    }
                }
            }
        }
        #endregion

        /// <summary>
        /// Swap the 4 quadrants, 1 <--> 3 and 2 <--> 4
        /// </summary>
        /// <param name="grid"></param>
        public static void Shift(Complex[,] grid)
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

        /// <summary>
        /// Swap the 4 quadrants, 1 <--> 3 and 2 <--> 4
        /// </summary>
        /// <param name="grid"></param>
        public static void Shift(double[,] grid)
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
