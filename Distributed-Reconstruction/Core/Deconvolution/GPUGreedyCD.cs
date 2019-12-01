using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

using ILGPU;
using ILGPU.AtomicOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

using ILGPU.Runtime.Cuda;


using static Core.Common;

namespace Core.Deconvolution
{
    public class GPUGreedyCD : Deconvolution.IDeconvolver, IDisposable
    {
        private static float GPUProximalOperator(float x, float lipschitz, float lambda, float alpha) => XMath.Max(x - lambda * alpha, 0.0f) / (lipschitz + lambda * (1 - alpha));

        #region GPU operations and GPU allocated data 
        private MemoryBuffer2D<float> xImageGPU;
        private MemoryBuffer2D<float> bMapGPU;
        private MemoryBuffer2D<float> aMapGPU;
        private MemoryBuffer2D<float> psf2GPU;

        private MemoryBuffer<float> lambdaAlpha;
        private MemoryBuffer<Pixel> maxPixelGPU;

        readonly Action<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>> shrink;
        readonly Action<ILGPU.Index, ArrayView2D<float>, ArrayView<Pixel>> updateX;
        readonly Action<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>> updateB;

        private void ShrinkGPU() => shrink(xImageGPU.Extent, xImageGPU.View, bMapGPU.View, aMapGPU.View, lambdaAlpha.View, maxPixelGPU.View);
        private void UpdateXGPU() => updateX(new ILGPU.Index(1), xImageGPU.View, maxPixelGPU.View);
        private void UpdateBGPU() => updateB(psf2GPU.Extent, bMapGPU.View, psf2GPU.View, maxPixelGPU.View);
        #endregion

        public bool RunsOnGPU { get; }
        readonly Context c;
        readonly Accelerator accelerator;
        readonly Rectangle totalSize;
        readonly float[,] psf2;
        float[,] aMap;
        readonly int batchIterations;

        public float MaxLipschitz { get; private set; }

        public static bool IsGPUSupported()
        {
            var c = new Context(ContextFlags.FastMath);
            
            var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
            var GPUSupported = gpuIds.Any();

            return GPUSupported;
        }

        public GPUGreedyCD(Rectangle totalSize, float[,] psf, int nrBatchIterations) :
            this(totalSize, psf, PSF.CalcPSFSquared(psf), nrBatchIterations)
        {

        }

        public GPUGreedyCD(Rectangle totalSize, float[,] psf, float[,] psfSquared, int nrBatchIterations)
        {
            this.totalSize = totalSize;
            psf2 = psfSquared;
            aMap = PSF.CalcAMap(psf, totalSize);
            MaxLipschitz = Residuals.GetMax(psfSquared);
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

            shrink = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>>(ShrinkKernel);
            updateX = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView2D<float>, ArrayView<Pixel>>(UpdateXKernel);
            updateB = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>>(UpdateBKernel);
        }



        #region IDeconvolver implementation
        public DeconvolutionResult DeconvolvePath(float[,] reconstruction, float[,] bMap, float lambdaMin, float lambdaFactor, float alpha, int maxPathIteration = 10, int maxIteration = 100, float epsilon = 0.0001f)
        {
            bool converged = false;
            AllocateGPU(reconstruction, bMap, lambdaMin, alpha);
            int totalIter = 0;
            var totalWatch = new Stopwatch();
            totalWatch.Start();
            for (int pathIter = 0; pathIter < maxPathIteration; pathIter++)
            {
                //set lambdap
                lambdaAlpha.CopyFrom(0.0f, new ILGPU.Index(0));
                ShrinkGPU();
                accelerator.Synchronize();
                var maxPixel = maxPixelGPU.GetAsArray()[0];
                var lambdaMax = 1 / alpha * maxPixel.AbsDiff;
                var lambdaCurrent = lambdaMax / lambdaFactor;
                lambdaCurrent = lambdaCurrent > lambdaMin ? lambdaCurrent : lambdaMin;

                maxPixelGPU.CopyFrom(new Pixel(0, -1, -1, 0), new ILGPU.Index(0));
                lambdaAlpha.CopyFrom(lambdaCurrent, new ILGPU.Index(0));

                Console.WriteLine("-----------------------------GPUGreedy with lambda " + lambdaCurrent + "------------------------");
                bool pathConverged = false;
                int i = 0;
                for (i = 0; i < maxIteration; i += batchIterations)
                {
                    DeconvolveBatch(batchIterations);
                    var lastPixel = maxPixelGPU.GetAsArray()[0];
                    if (lastPixel.AbsDiff < epsilon)
                    {
                        pathConverged = true;
                        break;
                    }
                }
                totalIter += i;

                converged = lambdaMin == lambdaCurrent & pathConverged;
                if (converged)
                    break;
            }
            totalWatch.Stop();

            xImageGPU.CopyTo(reconstruction, new Index2(0, 0), new Index2(0, 0), new Index2(reconstruction.GetLength(1), reconstruction.GetLength(0)));
            FreeGPU();

            return new DeconvolutionResult(converged, totalIter, totalWatch.Elapsed);
        }

        public DeconvolutionResult Deconvolve(float[,] reconstruction, float[,] bMap, float lambda, float alpha, int maxIteration, float epsilon=1e-4f)
        {
            AllocateGPU(reconstruction, bMap, lambda, alpha);

            var watch = new Stopwatch();
            watch.Start();
            bool converged = false;
            int iter = 0;
            while (!converged & iter < maxIteration)
            {
                DeconvolveBatch(batchIterations);
                var lastPixel = maxPixelGPU.GetAsArray()[0];
                if (lastPixel.AbsDiff < epsilon)
                {
                    converged = true;
                    break;
                }
                Console.WriteLine("iter\t" + (iter + batchIterations) + "\tcurrentUpdate\t" + lastPixel.AbsDiff);
                iter += batchIterations;
            }
            watch.Stop();

            xImageGPU.CopyTo(reconstruction, new Index2(0, 0), new Index2(0, 0), new Index2(reconstruction.GetLength(1), reconstruction.GetLength(0)));
            FreeGPU();

            return new DeconvolutionResult(converged, iter, watch.Elapsed);
        }

        public void ResetLipschitzMap(float[,] psf)
        {
            var psf2Local = PSF.CalcPSFSquared(psf);
            var maxFull = Residuals.GetMax(psf2Local);
            MaxLipschitz = maxFull;
            aMap = PSF.CalcAMap(psf, totalSize);

            var maxCut = Residuals.GetMax(psf2);
            for (int i = 0; i < psf2.GetLength(0); i++)
                for (int j = 0; j < psf2.GetLength(1); j++)
                    psf2[i, j] *= (maxFull / maxCut);
        }

        public float GetAbsMaxDiff(float[,] xImage, float[,] gradients, float lambda, float alpha)
        {
            var maxPixels = new float[xImage.GetLength(0)];
            Parallel.For(0, xImage.GetLength(0), (y) =>
            {
                var yLocal = y;

                var currentMax = 0.0f;
                for (int x = 0; x < xImage.GetLength(1); x++)
                {
                    var xLocal = x;
                    var L = aMap[yLocal, xLocal];
                    var old = xImage[yLocal, xLocal];
                    var xTmp = ElasticNet.ProximalOperator(old * L + gradients[y, x], L, lambda, alpha);
                    var xDiff = old - xTmp;

                    if (currentMax < Math.Abs(xDiff))
                        currentMax = Math.Abs(xDiff);
                }
                maxPixels[yLocal] = currentMax;
            });

            var maxPixel = 0.0f;
            for (int i = 0; i < maxPixels.Length; i++)
                if (maxPixel < maxPixels[i])
                    maxPixel = maxPixels[i];

            return maxPixel;
        }
        #endregion

        /// <summary>
        /// Make several deconvolution iterations without checking convergence
        /// </summary>
        /// <param name="batchIterations"></param>
        /// <returns></returns>
        private void DeconvolveBatch(int batchIterations)
        {
            for(int i = 0; i < batchIterations; i++)
            {
                ShrinkGPU();
                accelerator.Synchronize();
                UpdateXGPU();
                UpdateBGPU();
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
            var zeroIndex = new Index2(0, 0);
            var size = new Index2(xImage.GetLength(1), xImage.GetLength(0));
            xImageGPU = accelerator.Allocate<float>(size);
            xImageGPU.CopyFrom(xImage, zeroIndex, zeroIndex, size);

            var bMapSize = new Index2(bMap.GetLength(1), bMap.GetLength(0));
            bMapGPU = accelerator.Allocate<float>(bMapSize);
            bMapGPU.CopyFrom(bMap, zeroIndex, zeroIndex, bMapSize);

            aMapGPU = accelerator.Allocate<float>(size);
            aMapGPU.CopyFrom(aMap, zeroIndex, zeroIndex, size);

            var sizePSF = new Index2(psf2.GetLength(1), psf2.GetLength(0));
            psf2GPU = accelerator.Allocate<float>(sizePSF);
            psf2GPU.CopyFrom(psf2, zeroIndex, zeroIndex, sizePSF);

            lambdaAlpha = accelerator.Allocate<float>(2);
            lambdaAlpha.CopyFrom(lambda, new ILGPU.Index(0));
            lambdaAlpha.CopyFrom(alpha, new ILGPU.Index(1));

            maxPixelGPU = accelerator.Allocate<Pixel>(1);
        }

        /// <summary>
        /// Frees the buffers on the GPU
        /// </summary>
        private void FreeGPU()
        {
            xImageGPU.Dispose();
            bMapGPU.Dispose();
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
            ArrayView2D<float> bMap,
            ArrayView2D<float> aMap,
            ArrayView<float> lambdaAlpha,
            ArrayView<Pixel> output)
        {
            if (index.X == 0 & index.Y == 0)
                output[0].AbsDiff = 0;

            if (index.InBounds(xImage.Extent))
            {
                var xOld = xImage[index];
                var gradient = bMap[index];
                var lipschitz = aMap[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUProximalOperator(xOld * lipschitz + gradient, lipschitz, lambda, alpha);
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
            ILGPU.Index index,
            ArrayView2D<float> xImage,
            ArrayView<Pixel> pixel)
        {
            if (pixel[0].AbsDiff > 0)
            {
                xImage[pixel[0].X, pixel[0].Y] += pixel[0].Sign * pixel[0].AbsDiff;
            }
        }

        private static void UpdateBKernel(Index2 index,
            ArrayView2D<float> candidates,
            ArrayView2D<float> psf2,
            ArrayView<Pixel> pixel)
        {
            var indexCandidate = index.Add(new Index2(pixel[0].X, pixel[0].Y)).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(candidates.Extent) & pixel[0].AbsDiff > 0)
            {
                candidates[indexCandidate] -= (psf2[index] * pixel[0].Sign * pixel[0].AbsDiff);
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
