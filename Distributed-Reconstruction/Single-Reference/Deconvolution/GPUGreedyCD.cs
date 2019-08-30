using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

using ILGPU;
using ILGPU.AtomicOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using System.Linq;
using ILGPU.Runtime.Cuda;

using static Single_Reference.Common;

namespace Single_Reference.Deconvolution
{
    public class GPUGreedyCD : Deconvolution.IDeconvolver, IDisposable
    {
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => XMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region GPU allocated data buffers
        private MemoryBuffer2D<float> xImageGPU;
        private MemoryBuffer2D<float> candidatesGPU;
        private MemoryBuffer2D<float> aMapGPU;
        private MemoryBuffer2D<float> psf2GPU;

        private MemoryBuffer<float> lambdaAlpha;
        private MemoryBuffer<Pixel> maxPixelGPU;
        #endregion

        #region Loaded GPU kernels
        readonly Action<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>> shrink;
        readonly Action<Index, ArrayView2D<float>, ArrayView<Pixel>> updateX;
        readonly Action<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>> updateCandidates;
        #endregion

        public bool RunsOnGPU { get; }
        readonly Context c;
        readonly Accelerator accelerator;
        readonly Rectangle imageSection;
        readonly Rectangle psfSize;
        readonly Complex[,] psfCorrelation;
        readonly float[,] psf2;
        readonly float[,] aMap;
        readonly int batchIterations;

        public GPUGreedyCD(Rectangle totalSize, float[,] psf, int nrBatchIterations) :
            this(totalSize, totalSize, psf, PSF.CalcPaddedFourierCorrelation(psf, totalSize), PSF.CalcPSFSquared(psf), nrBatchIterations)
        {

        }

        public GPUGreedyCD(Rectangle totalSize, Rectangle imageSection, float[,] psf, int nrBatchIterations) :
            this(totalSize, imageSection, psf, PSF.CalcPaddedFourierCorrelation(psf, totalSize), PSF.CalcPSFSquared(psf), nrBatchIterations)
        {
            
        }

        public GPUGreedyCD(Rectangle totalSize, Rectangle imageSection, float[,] psf, Complex[,] psfCorrelation, float[,] psfSquared, int nrBatchIterations)
        {
            this.imageSection = imageSection;
            psfSize = new Rectangle(0, 0, psf.GetLength(0), psf.GetLength(1));
            this.psfCorrelation = psfCorrelation;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, totalSize, imageSection);
            batchIterations = nrBatchIterations;

            c = new Context(ContextFlags.FastMath);
            var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
            if (gpuIds.Any())
            {
                RunsOnGPU = true;
                accelerator = new CudaAccelerator(c, gpuIds.First().DeviceId);
            }
            else
            {
                Console.WriteLine("GPU vendor not supported. ILGPU switches to a !!!!VERY!!!! slow CPU implementation");
                RunsOnGPU = false;
                accelerator = new CPUAccelerator(c, 4);
            }

            shrink = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>>(ShrinkKernel);
            updateX = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<float>, ArrayView<Pixel>>(UpdateXKernel);
            updateCandidates = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>>(UpdateCandidatesKernel);
        }

        public bool DeconvolvePath(float[,] reconstruction, float[,] residuals, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            bool converged = false;
            var bMap = Residuals.CalcBMap(residuals, psfCorrelation, psfSize);
            for (int i = 0; i < bMap.GetLength(0); i++)
                for (int j = 0; j < bMap.GetLength(1); j++)
                    bMap[i, j] = bMap[i, j] / aMap[i, j];

            AllocateGPU(reconstruction, bMap, lambdaMin, alpha);

            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                //set lambdap
                lambdaAlpha.CopyFrom(0.0f, new Index(0));
                shrink(xImageGPU.Extent, xImageGPU.View, candidatesGPU.View, lambdaAlpha.View, maxPixelGPU.View);
                accelerator.Synchronize();
                var maxPixel = maxPixelGPU.GetAsArray()[0];
                var lambdaMax = 1 / alpha * maxPixel.AbsDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                maxPixelGPU.CopyFrom(new Pixel(0, -1, -1, 0), new Index(0));
                lambdaAlpha.CopyFrom(lambdaCurrent, new Index(0));

                Console.WriteLine("-----------------------------GPUGreedy with lambda " + lambdaCurrent + "------------------------");
                bool pathConverged = false;
                for (int i = 0; i < maxIteration; i += batchIterations)
                {
                    DeconvolutionBatchIterations(batchIterations);
                    var lastPixel = maxPixelGPU.GetAsArray()[0];
                    if (lastPixel.AbsDiff < epsilon)
                    {
                        pathConverged = true;
                        break;
                    }
                }

                converged = lambdaMin == lambdaCurrent & pathConverged;
                if (converged)
                    break;
            }

            xImageGPU.CopyTo(reconstruction, new Index2(0, 0), new Index2(0, 0), new Index2(reconstruction.GetLength(1), reconstruction.GetLength(0)));
            FreeGPU();
            return converged;
        }

        public bool Deconvolve(float[,] reconstruction, float[,] residuals, float lambda, float alpha, int iterations, float epsilon=1e-4f)
        {
            
            bool converged = false;
            var bMap = Residuals.CalcBMap(residuals, psfCorrelation, psfSize);
            for (int i = 0; i < bMap.GetLength(0); i++)
                for (int j = 0; j < bMap.GetLength(1); j++)
                    bMap[i, j] = bMap[i, j] / aMap[i, j];

            AllocateGPU(reconstruction, bMap, lambda, alpha);
            for (int i = 0; i < iterations; i+=batchIterations)
            {
                DeconvolutionBatchIterations(batchIterations);
                var lastPixel = maxPixelGPU.GetAsArray()[0];
                if(lastPixel.AbsDiff < epsilon)
                {
                    converged = true;
                    break;
                }
            }
            xImageGPU.CopyTo(reconstruction, new Index2(0, 0), new Index2(0, 0), new Index2(reconstruction.GetLength(1), reconstruction.GetLength(0)));
            FreeGPU();
            return converged;
        }

        /// <summary>
        /// Make several deconvolution iterations without checking convergence
        /// </summary>
        /// <param name="batchIterations"></param>
        /// <returns></returns>
        private void DeconvolutionBatchIterations(int batchIterations)
        {
            for(int i = 0; i < batchIterations; i++)
            {
                shrink(xImageGPU.Extent, xImageGPU.View, candidatesGPU.View, lambdaAlpha.View, maxPixelGPU.View);
                accelerator.Synchronize();

                updateX(new Index(1), xImageGPU.View, maxPixelGPU.View);
                updateCandidates(psf2GPU.Extent, candidatesGPU.View, aMapGPU.View, psf2GPU.View, maxPixelGPU.View);
                accelerator.Synchronize();
            }
        }

        /// <summary>
        /// Allocates and initializes buffers on the GPU
        /// </summary>
        /// <param name="xImage"></param>
        /// <param name="bMap">Gets modified by this method. output: bMap/aMap</param>
        /// <param name="lambda"></param>
        /// <param name="alpha"></param>
        private void AllocateGPU(float[,] xImage, float[,] bMap, float lambda, float alpha)
        {
            //TODO:windowing
            

            var zeroIndex = new Index2(0, 0);
            var size = new Index2(xImage.GetLength(1), xImage.GetLength(0));
            xImageGPU = accelerator.Allocate<float>(size);
            xImageGPU.CopyFrom(xImage, zeroIndex, zeroIndex, size);

            var bMapSize = new Index2(bMap.GetLength(1), bMap.GetLength(0));
            candidatesGPU = accelerator.Allocate<float>(bMapSize);
            candidatesGPU.CopyFrom(bMap, zeroIndex, zeroIndex, bMapSize);

            aMapGPU = accelerator.Allocate<float>(size);
            aMapGPU.CopyFrom(aMap, zeroIndex, zeroIndex, size);

            var sizePSF = new Index2(psf2.GetLength(1), psf2.GetLength(0));
            psf2GPU = accelerator.Allocate<float>(sizePSF);
            psf2GPU.CopyFrom(psf2, zeroIndex, zeroIndex, sizePSF);

            lambdaAlpha = accelerator.Allocate<float>(2);
            lambdaAlpha.CopyFrom(lambda, new Index(0));
            lambdaAlpha.CopyFrom(alpha, new Index(1));

            maxPixelGPU = accelerator.Allocate<Pixel>(1);
        }

        /// <summary>
        /// Frees the buffers on the GPU
        /// </summary>
        private void FreeGPU()
        {
            xImageGPU.Dispose();
            candidatesGPU.Dispose();
            aMapGPU.Dispose();
            psf2GPU.Dispose();

            lambdaAlpha.Dispose();
            maxPixelGPU.Dispose();
        }

        #region GPU pixel struct
        public struct Pixel : System.IEquatable<Pixel>
        {
            public float AbsDiff;
            public int Y;
            public int X;
            public int Sign;

            public Pixel(float abs, int x, int y, int sign)
            {
                AbsDiff = abs;
                Y = y;
                X = x;
                Sign = sign;
            }

            public bool Equals(Pixel other)
            {
                return AbsDiff == other.AbsDiff
                    & Y == other.Y
                    & X == other.X
                    & Sign == other.Sign;
            }
        }

        public struct MaxPixelOperation : IAtomicOperation<Pixel>
        {
            public Pixel Operation(Pixel current, Pixel value)
            {
                if (current.AbsDiff < value.AbsDiff)
                    return value;
                else
                    return current;
            }
        }

        public struct PixelCompareExchange : ICompareExchangeOperation<Pixel>
        {
            public Pixel CompareExchange(ref Pixel target, Pixel compare, Pixel value)
            {
                if (compare.AbsDiff != value.AbsDiff)
                {
                    var exchanged = Atomic.CompareExchange(ref target.AbsDiff, compare.AbsDiff, value.AbsDiff);
                    if (exchanged == compare.AbsDiff)
                    {
                        target.X = value.X;
                        target.Y = value.Y;
                        target.Sign = value.Sign;
                        return compare;
                    }
                }
                return target;
            }
        }
        #endregion

        #region GPU Kernels
        private static void ShrinkKernel(Index2 index,
            ArrayView2D<float> xImage,
            ArrayView2D<float> candidates,
            ArrayView<float> lambdaAlpha,
            ArrayView<Pixel> output)
        {
            if (index.X == 0 & index.Y == 0)
                output[0].AbsDiff = 0;

            if (index.InBounds(xImage.Extent))
            {
                var xOld = xImage[index];
                var xCandidate = candidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                var xAbsDiff = XMath.Abs(xNew - xOld);
                var xIndex = index.X;
                var yIndex = index.Y;
                var sign = XMath.Sign(xNew - xOld);

                var pix = new Pixel()
                {
                    AbsDiff = xAbsDiff,
                    X = xIndex,
                    Y = yIndex,
                    Sign = sign
                };
                Atomic.MakeAtomic(ref output[0], pix, new MaxPixelOperation(), new PixelCompareExchange());
            }
        }

        private static void UpdateXKernel(
            Index index,
            ArrayView2D<float> xImage,
            ArrayView<Pixel> pixel)
        {
            xImage[pixel[0].X, pixel[0].Y] += pixel[0].Sign * pixel[0].AbsDiff;
        }

        private static void UpdateCandidatesKernel(Index2 index,
            ArrayView2D<float> candidates,
            ArrayView2D<float> aMap,
            ArrayView2D<float> psf2,
            ArrayView<Pixel> pixel)
        {
            var indexCandidate = index.Add(new Index2(pixel[0].X, pixel[0].Y)).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(candidates.Extent))
            {
                candidates[indexCandidate] -= (psf2[index] * pixel[0].Sign * pixel[0].AbsDiff) / aMap[indexCandidate];
            }
        }
        #endregion

        #region Disposable implementation
        protected bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;
            if (disposing)
            {
                accelerator.Dispose();
                c.Dispose();
            }

            disposed = true;
        }

        #endregion
    }
}
