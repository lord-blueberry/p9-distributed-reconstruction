using System;
using System.Collections.Generic;
using System.Text;

using ILGPU;
using ILGPU.AtomicOperations;
using ILGPU.Lightning;
using ILGPU.ReductionOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.ShuffleOperations;
using System.Linq;
using ILGPU.Runtime.Cuda;

namespace Single_Reference.GPUDeconvolution
{
    public class GreedyCD : IDisposable
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

        public GreedyCD()
        {
            c = new Context(ContextFlags.FastMath);
            var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
            if (gpuIds.Any())
            {
                RunsOnGPU = true;
                accelerator = new CudaAccelerator(c, gpuIds.First().DeviceId);
            }  
            else
            {
                RunsOnGPU = false;
                accelerator = new CPUAccelerator(c, 4);
            }
                
            shrink = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>>(ShrinkKernel);
            updateX = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<float>, ArrayView<Pixel>>(UpdateXKernel);
            updateCandidates = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>>(UpdateCandidatesKernel);
        }

        public bool DeconvolvePath(float[,] xImage, float[,] bMap, float[,] aMap, float[,] psf2, float lambda, float alpha)
        {
            bool converged = false;
            AllocateGPU(xImage, bMap, aMap, psf2, lambda, alpha);

            FreeGPU();
            return converged;
        }

        public bool Deconvolve(float[,] xImage, float[,] bMap, float[,] aMap, float[,] psf2, float lambda, float alpha, float epsilon, int iterations, int batchIterations=500)
        {
            bool converged = false;
            AllocateGPU(xImage, bMap, aMap, psf2, lambda, alpha);
            for(int i = 0; i < iterations; i++)
            {
                DeconvolutionBatchIterations(batchIterations);
                var lastPixel = maxPixelGPU.GetAsArray()[0];
                if(lastPixel.AbsDiff < epsilon)
                {
                    converged = true;
                    break;
                }
            }
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
        /// <param name="aMap"></param>
        /// <param name="psf2"></param>
        /// <param name="lambda"></param>
        /// <param name="alpha"></param>
        private void AllocateGPU(float[,] xImage, float[,] bMap, float[,] aMap, float[,] psf2, float lambda, float alpha)
        {
            for (int i = 0; i < bMap.GetLength(0); i++)
                for (int j = 0; j < bMap.GetLength(1); j++)
                    bMap[i, j] = bMap[i, j] / aMap[i, j];

            var index0 = new Index2(0, 0);
            var size = new Index2(xImage.GetLength(1), xImage.GetLength(0));
            xImageGPU = accelerator.Allocate<float>(size);
            xImageGPU.CopyFrom(xImage, index0, index0, size);

            candidatesGPU = accelerator.Allocate<float>(size);
            candidatesGPU.CopyFrom(bMap, index0, index0, size);

            aMapGPU = accelerator.Allocate<float>(size);
            aMapGPU.CopyFrom(aMap, index0, index0, size);

            var sizePSF = new Index2(psf2.GetLength(1), psf2.GetLength(0));
            psf2GPU.CopyFrom(psf2, index0, index0, sizePSF);

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
            GC.SuppressFinalize(this);
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

        ~GreedyCD()
        {
            Dispose(false);
        }
        #endregion
    }
}
