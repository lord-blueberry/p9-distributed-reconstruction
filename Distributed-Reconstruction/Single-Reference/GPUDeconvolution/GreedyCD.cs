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

        readonly Context c;
        readonly Accelerator accelerator;

        public GreedyCD()
        {
            c = new Context(ContextFlags.FastMath);
            var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
            if (gpuIds.Any())
                accelerator = new CudaAccelerator(c, gpuIds.First().DeviceId);
            else
                accelerator = new CPUAccelerator(c, 4);

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
                this.DeconvolutionBatchIterations(batchIterations);
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

        #region slower kernels
        readonly Action<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<Pixel>> shrinkReduce;
        readonly Action<Index, ArrayView<int>, ArrayView2D<float>, ArrayView<Pixel>, ArrayView<Pixel>> reduceAndUpdateX;
        readonly Action<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>> updateBKernelV0;
        readonly Action<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>> updateBKernelV1;

        private static void ShrinkReduceKernel(
            GroupedIndex grouped,
            ArrayView2D<float> xImage,
            ArrayView2D<float> bMap,
            ArrayView2D<float> aMap,
            ArrayView<float> lambdaAlpha,
            ArrayView<Pixel> maxPixels)
        {
            var gridIdx = grouped.GridIdx;
            var threadID = grouped.GroupIdx;
            var warpIdx = threadID / Warp.WarpSize;

            var lambda = lambdaAlpha[0];
            var alpha = lambdaAlpha[1];

            //shared memory for warp reduce
            int warpCount = 32;
            var sharedXDiff = SharedMemory.Allocate<float>(warpCount);
            var sharedXAbsDiff = SharedMemory.Allocate<float>(warpCount);
            var sharedXIndex = SharedMemory.Allocate<int>(warpCount);
            var sharedYIndex = SharedMemory.Allocate<int>(warpCount);

            //assign y indices to the different threadgroups. y seems to be the major index in ILGPU
            var yCount = xImage.Extent.Y / (float)(Grid.DimensionX);
            var yIdx = (int)(gridIdx * yCount);
            var yIdxEnd = (int)((gridIdx + 1) * yCount);
            yIdxEnd = gridIdx + 1 == Grid.DimensionX ? xImage.Extent.Y : yIdxEnd;
            var fromPixel = new Index2(0, yIdx).ComputeLinearIndex(xImage.Extent);
            var toPixel = new Index2(xImage.Extent.X, yIdxEnd - 1).ComputeLinearIndex(xImage.Extent);

            //create linear views per group
            var linX = xImage.AsLinearView();
            var linB = bMap.AsLinearView();
            var linA = aMap.AsLinearView();

            //assign consecutive pixels to threads in a group.
            var pixelCount = (toPixel - fromPixel) / (float)(Group.Dimension.X);
            var pixelIdx = (int)(threadID * pixelCount) + fromPixel;
            var pixelEnd = (int)((threadID + 1) * pixelCount) + fromPixel;
            pixelEnd = threadID + 1 == Group.DimensionX ? toPixel : pixelEnd;

            //shrink and save max of the assigned pixels
            float xDiff = 0.0f;
            float xAbsDiff = float.MinValue;
            int xIndex = -1;
            int yIndex = -1;
            for (int i = pixelIdx; i < pixelEnd; i++)
            {
                var xOld = linX[i];
                var xNew = xOld + linB[i] / linA[i];
                xNew = GPUShrinkElasticNet(xNew, lambda, alpha);
                var tmpDiff = xNew - xOld;

                if (xAbsDiff < XMath.Abs(tmpDiff))
                {
                    xDiff = tmpDiff;
                    xAbsDiff = XMath.Abs(tmpDiff);
                    var recIndex = Index2.ReconstructIndex(i, xImage.Extent);
                    xIndex = recIndex.X;
                    yIndex = recIndex.Y;
                }
            }

            Warp.Barrier();
            for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                var oXDiff = Warp.ShuffleDown(xDiff, offset);
                var oXIndex = Warp.ShuffleDown(xIndex, offset);
                var oYIndex = Warp.ShuffleDown(yIndex, offset);
                if (xAbsDiff < oXAbsDiff)
                {
                    xAbsDiff = oXAbsDiff;
                    xDiff = oXDiff;
                    xIndex = oXIndex;
                    yIndex = oYIndex;
                }
            }
            if (Warp.IsFirstLane)
            {
                sharedXDiff[warpIdx] = xDiff;
                sharedXAbsDiff[warpIdx] = xAbsDiff;
                sharedXIndex[warpIdx] = xIndex;
                sharedYIndex[warpIdx] = yIndex;
            }
            Group.Barrier();

            //warp 0 reduce of shared memory
            var maxShared = Group.Dimension.X / Warp.WarpSize;
            if (warpIdx == 0 & Warp.LaneIdx < maxShared)
            {
                xDiff = sharedXDiff[Warp.LaneIdx];
                xAbsDiff = sharedXAbsDiff[Warp.LaneIdx];
                xIndex = sharedXIndex[Warp.LaneIdx];
                yIndex = sharedYIndex[Warp.LaneIdx];

                //warp reduce
                for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
                {
                    var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                    var oXDiff = Warp.ShuffleDown(xDiff, offset);
                    var oXIndex = Warp.ShuffleDown(xIndex, offset);
                    var oYIndex = Warp.ShuffleDown(yIndex, offset);
                    if (xAbsDiff < oXAbsDiff)
                    {
                        xAbsDiff = oXAbsDiff;
                        xDiff = oXDiff;
                        xIndex = oXIndex;
                        yIndex = oYIndex;
                    }
                }

                if (Warp.IsFirstLane)
                {
                    var pixel = new Pixel() {
                        AbsDiff = xAbsDiff,
                        X = xIndex,
                        Y = yIndex,
                        Sign = XMath.Sign(xDiff)
                    };

                    maxPixels[gridIdx] = pixel;
                }
            }

            //if accelerator has no warp, reduce the shared memory the old fashioned way
            if (Warp.WarpSize == 1)
            {
                if (threadID == 0)
                {
                    for (int i = 0; i < maxShared; i++)
                    {
                        if (xAbsDiff < sharedXAbsDiff[i])
                        {
                            xDiff = sharedXDiff[i];
                            xAbsDiff = sharedXAbsDiff[i];
                            xIndex = sharedXIndex[i];
                            yIndex = sharedYIndex[i];
                        }
                    }
                    var pixel = new Pixel()
                    {
                        AbsDiff = xAbsDiff,
                        X = xIndex,
                        Y = yIndex,
                        Sign = XMath.Sign(xDiff)
                    };

                    maxPixels[gridIdx] = pixel;
                }
            }
        }

        private static void ReduceAndUpdate(
            Index threadIndex,
            ArrayView<int> totalThreads,
            ArrayView2D<float> xImage,
            ArrayView<Pixel> maxPixels,
            ArrayView<Pixel> maxPixelOut)
        {
            var itemCount = maxPixels.Extent.X / (float)(totalThreads[0]);
            var itemIdx = (int)(threadIndex * itemCount);
            var itemEnd = (int)((threadIndex + 1) * itemCount);
            itemEnd = threadIndex + 1 == totalThreads[0] ? maxPixels.Extent.X : itemEnd;

            var tempSign = maxPixels[itemIdx].Sign;
            var tempAbs = maxPixels[itemIdx].AbsDiff;
            var tempXIndex = maxPixels[itemIdx].X;
            var tempYIndex = maxPixels[itemIdx].Y;
            for (int i = itemIdx + 1; i < itemEnd; i++)
                if (tempAbs < maxPixels[i].AbsDiff)
                {
                    tempSign = maxPixels[i].Sign;
                    tempAbs = maxPixels[i].AbsDiff;
                    tempXIndex = maxPixels[i].X;
                    tempYIndex = maxPixels[i].Y;
                }

            Warp.Barrier();
            for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                var oXAbsDiff = Warp.ShuffleDown(tempAbs, offset);
                var oXDiff = Warp.ShuffleDown(tempSign, offset);
                var oXIndex = Warp.ShuffleDown(tempXIndex, offset);
                var oYIndex = Warp.ShuffleDown(tempYIndex, offset);
                if (tempAbs < oXAbsDiff)
                {
                    tempAbs = oXAbsDiff;
                    tempSign = oXDiff;
                    tempXIndex = oXIndex;
                    tempYIndex = oYIndex;
                }
            }

            //if there is no warp, do it single threaded
            if (Warp.WarpSize == 1 & threadIndex == 0)
            {
                for (int i = itemEnd; i < maxPixels.Extent.X; i++)
                {
                    if (tempAbs < maxPixels[i].AbsDiff)
                    {
                        tempSign = maxPixels[i].Sign;
                        tempAbs = maxPixels[i].AbsDiff;
                        tempXIndex = maxPixels[i].X;
                        tempYIndex = maxPixels[i].Y;
                    }
                }
            }

            if (threadIndex == 0)
            {
                xImage[tempXIndex, tempYIndex] += (tempSign * tempAbs);
                var outPixel = new Pixel()
                {
                    AbsDiff = tempAbs,
                    X = tempXIndex,
                    Y = tempYIndex,
                    Sign = tempSign
                };
                maxPixelOut[0] = outPixel;
            }
        }

        private static void UpdateBKernelV1(
            GroupedIndex grouped,
            ArrayView2D<float> bMap,
            ArrayView2D<float> psf2,
            ArrayView<Pixel> maxPixel)
        {
            var gridIdx = grouped.GridIdx;
            var threadID = grouped.GroupIdx;

            //assign y indices to the different threadgroups. y seems to be the major index in ILGPU
            var yCount = psf2.Extent.Y / (float)(Grid.DimensionX);
            var yIdx = (int)(gridIdx * yCount);
            var yIdxEnd = (int)((gridIdx + 1) * yCount);
            yIdxEnd = gridIdx + 1 == Grid.DimensionX ? psf2.Extent.Y : yIdxEnd;
            var fromPixel = new Index2(0, yIdx).ComputeLinearIndex(psf2.Extent);
            var toPixel = new Index2(psf2.Extent.X, yIdxEnd - 1).ComputeLinearIndex(psf2.Extent);

            var linPSF = psf2.AsLinearView();

            //assign consecutive pixels to threads in a group.
            var pixelCount = (toPixel - fromPixel) / (float)(Group.Dimension.X);
            var pixelIdx = (int)(threadID * pixelCount) + fromPixel;
            var pixelEnd = (int)((threadID + 1) * pixelCount) + fromPixel;
            pixelEnd = threadID + 1 == Group.DimensionX ? toPixel : pixelEnd;

            var offset = new Index2(maxPixel[0].X, maxPixel[0].Y).Subtract(psf2.Extent / 2);
            for (int i = pixelIdx; i < pixelEnd; i++)
            {
                var p = linPSF[i];
                var psfIndex = Index2.ReconstructIndex(i, psf2.Extent);
                var indexBMap = psfIndex.Add(offset);
                if (indexBMap.InBounds(bMap.Extent))
                {
                    bMap[indexBMap] -= (p * maxPixel[0].Sign * maxPixel[0].AbsDiff);
                }
            }
        }

        private static void UpdateBKernelV0(
            Index2 index,
            ArrayView2D<float> bMap,
            ArrayView2D<float> psf2,
            ArrayView<Pixel> maxPixel)
        {
            var indexCandidate = index.Add(new Index2(maxPixel[0].X, maxPixel[0].Y)).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(bMap.Extent))
            {
                bMap[indexCandidate] -= (psf2[index] * maxPixel[0].Sign * maxPixel[0].AbsDiff);
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
