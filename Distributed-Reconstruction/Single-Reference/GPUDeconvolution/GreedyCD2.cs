using System;

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
    public class GreedyCD2
    {
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => XMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region atomic MaxPixel operation
        public struct MaxPixel : System.IEquatable<MaxPixel>
        {
            public float AbsDiff;
            public int Y;
            public int X;
            public int Sign;

            public bool Equals(MaxPixel other)
            {
                return AbsDiff == other.AbsDiff
                    & Y == other.Y
                    & X == other.X
                    & Sign == other.Sign;
            }
        }

        public struct MaxPixelOperation : IAtomicOperation<MaxPixel>
        {
            public MaxPixel Operation(MaxPixel current, MaxPixel value)
            {
                if (current.AbsDiff < value.AbsDiff)
                    return value;
                else
                    return current;
            }
        }

        public struct MaxPixelCompareExchange : ICompareExchangeOperation<MaxPixel>
        {
            public MaxPixel CompareExchange(ref MaxPixel target, MaxPixel compare, MaxPixel value)
            {

                if (target.AbsDiff == compare.AbsDiff)
                {
                    target.AbsDiff = value.AbsDiff;
                    target.X = value.X;
                    target.Y = value.Y;
                    target.Sign = value.Sign;
                    return compare;
                }
                return target;
            }
        }
        #endregion

        #region kernels

        private static void ShrinkReduceKernel(
            GroupedIndex grouped,
            ArrayView2D<float> xImage,
            ArrayView2D<float> bMap,
            ArrayView2D<float> aMap,
            ArrayView<float> lambdaAlpha,
            ArrayView<float> xDiffOut,
            ArrayView<float> xAbsDiffOut,
            ArrayView<int> xIndexOut,
            ArrayView<int> yIndexOut)
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
                //Warp.Barrier();
            }
            //Warp.Barrier();
            if (Warp.IsFirstLane)
            {
                sharedXDiff[warpIdx] = xDiff;
                sharedXAbsDiff[warpIdx] = xAbsDiff;
                sharedXIndex[warpIdx] = xIndex;
                sharedYIndex[warpIdx] = yIndex;

                //debugOut[gridIdx, warpIdx] = xAbsDiff;
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

                //Warp.Barrier();
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
                    //Warp.Barrier();
                }

                if (Warp.IsFirstLane)
                {
                    xDiffOut[gridIdx] = xDiff;
                    xAbsDiffOut[gridIdx] = xAbsDiff;
                    xIndexOut[gridIdx] = xIndex;
                    yIndexOut[gridIdx] = yIndex;
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
                    xDiffOut[gridIdx] = xDiff;
                    xAbsDiffOut[gridIdx] = xAbsDiff;
                    xIndexOut[gridIdx] = xIndex;
                    yIndexOut[gridIdx] = yIndex;
                }
            }
        }

        private static void ReduceAndUpdate(
            Index threadIndex,
            ArrayView<int> totalThreads,
            ArrayView2D<float> xImage,
            ArrayView<float> xDiff,
            ArrayView<float> xAbsDiff,
            ArrayView<int> xIndex,
            ArrayView<int> yIndex,
            ArrayView<float> maxDiffOut,
            ArrayView<int> indicesOut)
        {
            var itemCount = xDiff.Extent.X / (float)(totalThreads[0]);
            var itemIdx = (int)(threadIndex * itemCount);
            var itemEnd = (int)((threadIndex + 1) * itemCount);
            itemEnd = threadIndex + 1 == totalThreads[0] ? xDiff.Extent.X : itemEnd;

            var tempX = xDiff[itemIdx];
            var tempAbs = xAbsDiff[itemIdx];
            var tempXIndex = xIndex[itemIdx];
            var tempYIndex = yIndex[itemIdx];
            for (int i = itemIdx + 1; i < itemEnd; i++)
                if (tempAbs < xAbsDiff[i])
                {
                    tempX = xDiff[i];
                    tempAbs = xAbsDiff[i];
                    tempXIndex = xIndex[i];
                    tempYIndex = yIndex[i];
                }

            Warp.Barrier();
            for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                var oXAbsDiff = Warp.ShuffleDown(tempAbs, offset);
                var oXDiff = Warp.ShuffleDown(tempX, offset);
                var oXIndex = Warp.ShuffleDown(tempXIndex, offset);
                var oYIndex = Warp.ShuffleDown(tempYIndex, offset);
                if (tempAbs < oXAbsDiff)
                {
                    tempAbs = oXAbsDiff;
                    tempX = oXDiff;
                    tempXIndex = oXIndex;
                    tempYIndex = oYIndex;
                }
                //Warp.Barrier();
            }

            if(Warp.WarpSize == 1 & threadIndex == 0)
            {
                for(int i = itemEnd; i < xDiff.Extent.X;i++)
                {
                    if (tempAbs < xAbsDiff[i])
                    {
                        tempX = xDiff[i];
                        tempAbs = xAbsDiff[i];
                        tempXIndex = xIndex[i];
                        tempYIndex = yIndex[i];
                    }
                }
            }

            if (threadIndex == 0)
            {
                xImage[tempXIndex, tempYIndex] += tempX;
                maxDiffOut[0] = tempX;
                indicesOut[0] = tempXIndex;
                indicesOut[1] = tempYIndex;
            }
        }

        private static void UpdateBKernelV1(
            GroupedIndex grouped,
            ArrayView2D<float> bMap,
            ArrayView2D<float> psf2,
            ArrayView<float> maxDiff,
            ArrayView<int> maxIndices)
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

            var offset = new Index2(maxIndices[0], maxIndices[1]).Subtract(psf2.Extent / 2);
            for (int i = pixelIdx; i < pixelEnd; i++)
            {
                var p = linPSF[i];
                var psfIndex = Index2.ReconstructIndex(i, psf2.Extent);
                var indexBMap = psfIndex.Add(offset);
                if(indexBMap.InBounds(bMap.Extent))
                {
                    bMap[indexBMap] -= (p * maxDiff[0]);
                }
            }
        }

        private static void UpdateBKernelV0(
            Index2 index,
            ArrayView2D<float> bMap,
            ArrayView2D<float> psf2,
            ArrayView<float> maxDiff,
            ArrayView<int> maxIndices)
        {
            var indexCandidate = index.Add(new Index2(maxIndices[0], maxIndices[1])).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(bMap.Extent))
            {
                bMap[indexCandidate] -= (psf2[index] * maxDiff[0]);
            }
        }
        #endregion


        private static void Iteration(Accelerator accelerator, float[,] xImageIn, float[,] bMapIn, float[,] aMapIn, float[,] psf2In, float lambda, float alpha)
        {
            var maxGroups = accelerator.MaxNumThreads / accelerator.MaxNumThreadsPerGroup;
            var groupThreadIdx = new GroupedIndex(maxGroups, accelerator.MaxNumThreadsPerGroup);
            maxGroups = 2;
            groupThreadIdx = new GroupedIndex(maxGroups, 4);
            var nextPower2 = 1 << (63 - CountLeadingZeroBits((UInt64)maxGroups));    //calculate the next smallest power of 2 value for the group size. Used for reduceAndUpdate

            var shrinkReduce = accelerator.LoadStreamKernel<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>>(ShrinkReduceKernel);
            var reduceAndUpdateX = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, ArrayView<float> , ArrayView<int>>(ReduceAndUpdate);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateBKernelV0);
            var updateBKernel = accelerator.LoadStreamKernel<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateBKernelV1);


            var size = new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1));
            var psfSize = new Index2(psf2In.GetLength(0), psf2In.GetLength(1));

            using (var xImage = accelerator.Allocate<float>(size))
            using (var bMap = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))

            using (var maxDiff = accelerator.Allocate<float>(maxGroups))
            using (var maxAbsDiff = accelerator.Allocate<float>(maxGroups))
            using (var xIndex = accelerator.Allocate<int>(maxGroups))
            using (var yIndex = accelerator.Allocate<int>(maxGroups))

            using (var maxPixel = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            using (var maxReduceThreads = accelerator.Allocate<int>(1))
            {
                xImage.CopyFrom(xImageIn, new Index2(0, 0), new Index2(0, 0), new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1)));
                bMap.CopyFrom(bMapIn, new Index2(0, 0), new Index2(0, 0), new Index2(bMapIn.GetLength(0), bMapIn.GetLength(1)));
                aMap.CopyFrom(aMapIn, new Index2(0, 0), new Index2(0, 0), new Index2(aMapIn.GetLength(0), aMapIn.GetLength(1)));
                psf2.CopyFrom(psf2In, new Index2(0, 0), new Index2(0, 0), new Index2(psf2In.GetLength(0), psf2In.GetLength(1)));

                lambdaAlpha.CopyFrom(lambda, new Index(0));
                lambdaAlpha.CopyFrom(alpha, new Index(1));
                maxReduceThreads.CopyFrom(nextPower2, new Index(0));
                Console.WriteLine("Start");
                var watch = new System.Diagnostics.Stopwatch();
                watch.Start();
                for (int i = 0; i < 1000; i++)
                {
                    shrinkReduce(groupThreadIdx, xImage.View, bMap.View, aMap.View, lambdaAlpha.View, maxDiff.View, maxAbsDiff.View, xIndex.View, yIndex.View);
                    accelerator.Synchronize();
                    /*
                    var maxDiffT = 0.0f;
                    var maxAbsDiffT = 0.0f;
                    var xIndexT = -1;
                    var yIndexT = -1;
                    var t0 = maxAbsDiff.GetAsArray();
                    var t1 = maxDiff.GetAsArray();
                    var t2 = xIndex.GetAsArray();
                    var t3 = yIndex.GetAsArray();
                    for (int j = 0; j < t0.Length; j++)
                        if (maxAbsDiffT < t0[j])
                        {
                            maxDiffT = t1[j];
                            maxAbsDiffT = t0[j];
                            xIndexT = t2[j];
                            yIndexT = t3[j];
                        }
                        */
                    reduceAndUpdateX(new Index(nextPower2), maxReduceThreads.View, xImage.View, maxDiff.View, maxAbsDiff.View, xIndex.View, yIndex.View, maxPixel.View, maxIndices.View);
                    accelerator.Synchronize();
                    updateCandidatesKernel(psfSize, bMap.View, psf2.View, maxPixel.View, maxIndices.View);
                    //updateBKernel(groupThreadIdx, bMap.View, psf2.View, maxPixel.View, maxIndices.View);
                    accelerator.Synchronize();

                    //Console.WriteLine("iteration " + i);
                }

                watch.Stop();
                Console.WriteLine(watch.Elapsed);
                var bla = Console.ReadLine();
                var xOutput = xImage.GetAsArray();
                var candidate = bMap.GetAsArray();
                //FitsIO.Write(CopyToImage(xOutput, size), "xImageGPU.fits");
                //FitsIO.Write(CopyToImage(candidate, size), "candidateGPU.fits");
            }
        }



        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf, double lambda, double alpha)
        {
            using (var context = new Context(ContextFlags.FastMath, ILGPU.IR.Transformations.OptimizationLevel.Release))
            {
                var psf2 = CommonMethods.PSF.CalcPSFSquared(xImage, psf);
                var aMap = CommonMethods.PSF.CalcAMap(xImage, psf);
                var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
                if (gpuIds.Any())
                {
                    using (var accelerator = new CudaAccelerator(context, gpuIds.First().DeviceId))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        Iteration(accelerator, ToFloat(xImage), ToFloat(b), ToFloat(aMap), ToFloat(psf2), (float)lambda, (float)alpha);
                    }
                }
                else
                {
                    using (var accelerator = new CPUAccelerator(context, 4))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        Iteration(accelerator, ToFloat(xImage), ToFloat(b), ToFloat(aMap), ToFloat(psf2), (float)lambda, (float)alpha);
                    }
                }
            }

            return true;
        }


        private static float[,] ToFloat(double[,] img)
        {
            var output = new float[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                    output[i, j] = (float)img[i, j];
            return output;
        }

        private static int CountLeadingZeroBits(UInt64 input)
        {
            if (input == 0) return 64;

            UInt64 n = 1;

            if ((input >> 32) == 0) { n = n + 32; input = input << 32; }
            if ((input >> 48) == 0) { n = n + 16; input = input << 16; }
            if ((input >> 56) == 0) { n = n + 8; input = input << 8; }
            if ((input >> 60) == 0) { n = n + 4; input = input << 4; }
            if ((input >> 62) == 0) { n = n + 2; input = input << 2; }
            n = n - (input >> 63);

            return (int)n;
        }


        public static double[,] CopyToImage(float[] img, Index2 size)
        {
            var output = new double[size.Y, size.X];
            var index = 0;
            for (int y = 0; y < size.Y; y++)
            {
                for (int x = 0; x < size.X; x++)
                    output[x, y] = img[index + x];
                index += size.X;
            }

            return output;
        }
    }
}
