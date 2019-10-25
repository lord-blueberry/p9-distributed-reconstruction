using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using static Single_Reference.Common;

namespace Single_Reference
{
    /// <summary>
    /// Faster implementation of a padded convolution in fourier space.
    /// </summary>
    public class PaddedConvolver :IDisposable
    {
        readonly FFT fft;
        readonly Rectangle kernelSize;
        Complex[,] kernel;

        public PaddedConvolver(Complex[,] kernel, Rectangle kernelSize)
        {
            fft = new FFT(kernel.GetLength(0), kernel.GetLength(1));
            this.kernel = kernel;
            this.kernelSize = kernelSize;
        }

        public float[,] Convolve(float[,] image)
        {
            InsertImage(image);
            fft.Forward();
            for (int i = 0; i < kernel.GetLength(0); i++)
                for (int j = 0; j < kernel.GetLength(1); j++)
                    fft.FourierBuffer[i, j] *= kernel[i, j];
            fft.Backward();

            var output = new float[image.GetLength(0), image.GetLength(1)];
            var yHalf = kernelSize.YExtent() / 2;
            var xHalf = kernelSize.XExtent() / 2;
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(0); j++)
                    output[i, j] = (float)(fft.ImageBuffer[i + yHalf, j + xHalf].Real / kernel.Length);
            return output;
        }

        public void ConvolveInPlace(float[,] image)
        {
            InsertImage(image);
            fft.Forward();
            for (int i = 0; i < kernel.GetLength(0); i++)
                for (int j = 0; j < kernel.GetLength(1); j++)
                    fft.FourierBuffer[i, j] *= kernel[i, j]; 
            fft.Backward();

            var yHalf = kernelSize.YExtent() / 2;
            var xHalf = kernelSize.XExtent() / 2;
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(0); j++)
                    image[i, j] = (float)(fft.ImageBuffer[i + yHalf, j + xHalf].Real / kernel.Length);
        }

        private void InsertImage(float[,] image)
        {
            var yHalf = kernelSize.YExtent() / 2;
            var xHalf = kernelSize.XExtent() / 2;
            var imgDimensions = new Rectangle(yHalf, xHalf, image.GetLength(0) + yHalf, image.GetLength(1) + xHalf);

            for (int i = 0; i < kernel.GetLength(0); i++)
                for (int j = 0; j < kernel.GetLength(1); j++)
                    if (imgDimensions.PointInRectangle(i, j))
                    {
                        fft.ImageBuffer[i, j] = image[i - yHalf, j - xHalf];
                    }
                    else
                    {
                        fft.ImageBuffer[i, j] = Complex.Zero;
                    }
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
                }


                disposedValue = true;
            }
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
    }
}
