﻿using System;

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

        private struct MaxPixel
        {
            int y;
            int x;
            float xDiff;
            float xAbsDiff; 
        }

        #region kernels
        private static void TestShuffle(Index index, ArrayView<int> output, ArrayView<int> output2)
        {
            var local = index;
            var l2 = index * 10;
            /*
            var d = XMath.Max(Warp.ShuffleDown(local, Warp.WarpSize / 2), local);
            d = XMath.Max(Warp.ShuffleDown(d, Warp.WarpSize / 4), d);
            d = XMath.Max(Warp.ShuffleDown(d, Warp.WarpSize / 8), d);
            d = XMath.Max(Warp.ShuffleDown(d, Warp.WarpSize / 16), d);
            d = XMath.Max(Warp.ShuffleDown(d, Warp.WarpSize / 32), d);*/
           
            var d2 = local;
            Warp.Barrier();
            for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                //d2 = XMath.Max(Warp.ShuffleDown(d2, offset), d2);
                var tmp = Warp.ShuffleDown(d2, offset);
                var tmp1 = Warp.ShuffleDown(l2, offset);
                if (d2 < tmp)
                {
                    d2 = tmp;
                    l2 = tmp1;
                }
                    
                Warp.Barrier();
            }
            output[local] = d2;
            output2[local] = l2;
        }


        private static void TestKernel(
            GroupedIndex grouped,
            ArrayView2D<float> xImage)
        {
            var gridIdx = grouped.GridIdx;
            var threadID = grouped.GroupIdx;
            var warpIdx = threadID / Warp.WarpSize;

            var rowCount = (xImage.Extent.Y + Grid.Dimension.X - 1) / Grid.Dimension.X;
            var rowIdx = gridIdx * rowCount;
            var rowExtent = XMath.Min(rowIdx + rowCount, xImage.Extent.Y) - rowIdx;

            var subView = xImage.GetSubView(new Index2(rowIdx, 0), new Index2(1, xImage.Extent.Y));

            var linView = subView.AsLinearView();

            var superLin = xImage.AsLinearView();
            var fromIndex = new Index2(0, rowIdx);
            var toIndex = new Index2(xImage.Extent.X, rowExtent + rowIdx -1 );

            var linF = fromIndex.ComputeLinearIndex(xImage.Extent);
            var linT = toIndex.ComputeLinearIndex(xImage.Extent);

        }

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
            var yCount = (xImage.Extent.Y + Grid.Dimension.X - 1) / Grid.Dimension.X;
            var yIdx = gridIdx * yCount;
            var yIdxEnd = XMath.Min(yIdx + yCount, xImage.Extent.Y);
            
            //create linear views per group
            var linX = xImage.AsLinearView();
            var linB = bMap.AsLinearView();
            var linA = aMap.AsLinearView();
            var fromPixel = new Index2(0, yIdx).ComputeLinearIndex(xImage.Extent);
            var toPixel = new Index2(xImage.Extent.X, yIdxEnd - 1).ComputeLinearIndex(xImage.Extent);

            //assign consecutive pixels to threads in a group.
            var pixelCount = (toPixel - fromPixel + Group.Dimension.X - 1) / Group.Dimension.X;
            var pixelIdx = threadID * pixelCount + fromPixel;
            var pixelEnd = XMath.Min(pixelIdx + pixelCount, toPixel);

            //shrink and save max of the assigned pixels
            float xDiff = 0.0f;
            float xAbsDiff = float.MinValue;
            int xIndex = -1;
            int yIndex = -1;
            for(int i = pixelIdx; i < pixelEnd; i++)
            {
                var xOld = linX[i];
                var xNew = xOld + linB[i] / linA[i];
                xNew = GPUShrinkElasticNet(xNew, lambda, alpha);
                var tmpDiff = xOld - xNew;
                
                if(xAbsDiff < XMath.Abs(tmpDiff))
                {
                    xDiff = tmpDiff;
                    xAbsDiff = XMath.Abs(tmpDiff);
                    var recIndex = Index2.ReconstructIndex(i, xImage.Extent);
                    xIndex = recIndex.X;
                    yIndex = recIndex.Y;
                }
            }

            Warp.Barrier();
            //warp reduce
            for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                if (oXAbsDiff > xAbsDiff)
                {
                    xAbsDiff = oXAbsDiff;
                    xDiff = Warp.ShuffleDown(xDiff, offset);
                    xIndex = Warp.ShuffleDown(xIndex, offset);
                    yIndex = Warp.ShuffleDown(yIndex, offset);
                }
                Warp.Barrier();
            }
            Warp.Barrier();
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
            if (warpIdx == 0 & Warp.LaneIdx + 1 < maxShared)
            {
                xDiff = sharedXIndex[Warp.LaneIdx];
                xAbsDiff = sharedXAbsDiff[Warp.LaneIdx];
                xIndex = sharedXIndex[Warp.LaneIdx];
                yIndex = sharedYIndex[Warp.LaneIdx];
                
                //warp reduce
                for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
                {
                    var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                    if (xAbsDiff < oXAbsDiff)
                    {
                        xAbsDiff = oXAbsDiff;
                        xDiff = Warp.ShuffleDown(xDiff, offset);
                        xIndex = Warp.ShuffleDown(xIndex, offset);
                        yIndex = Warp.ShuffleDown(yIndex, offset);
                        
                    }
                    Warp.Barrier();
                }
                Warp.Barrier();
                if (Warp.IsFirstLane)
                {
                    xDiffOut[gridIdx] = xDiff;
                    xAbsDiffOut[gridIdx] = xAbsDiff;
                    xIndexOut[gridIdx] = xIndex;
                    yIndexOut[gridIdx] = yIndex;
                }
            }

            //if accelerator has no warp, reduce the shared memory the old fashioned way
            if(Warp.WarpSize >= 1)
            {
                if (threadID == 0)
                {
                    for(int i = 0; i < maxShared; i++)
                    {
                        if(xAbsDiff < sharedXAbsDiff[i])
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

        private static void ShrinkKernel(
             GroupedIndex grouped,
             ArrayView2D<float> xImage,
             ArrayView2D<float> bMap,
             ArrayView2D<float> aMap,
             ArrayView<float> lambdaAlpha,
             ArrayView2D<float> xDiffOut,
             ArrayView2D<float> xAbsDiffOut,
             ArrayView2D<int> xIndexOut,
             ArrayView2D<int> yIndexOut)
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
            var yIdxEnd = (int)((gridIdx+1) * yCount);
            yIdxEnd = gridIdx + 1 == Grid.DimensionX ? xImage.Extent.Y : yIdxEnd;

            //create linear views per group
            var linX = xImage.AsLinearView();
            var linB = bMap.AsLinearView();
            var linA = aMap.AsLinearView();
            var fromPixel = new Index2(0, yIdx).ComputeLinearIndex(xImage.Extent);
            var toPixel = new Index2(xImage.Extent.X, yIdxEnd - 1).ComputeLinearIndex(xImage.Extent);

            //assign consecutive pixels to threads in a group.
            var pixelCount = (toPixel - fromPixel) / (float)(Group.Dimension.X);
            var pixelIdx = (int)(threadID * pixelCount) + fromPixel;
            var pixelEnd = (int)((threadID+1) * pixelCount) + fromPixel;
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
                var tmpDiff = xOld - xNew;

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
            var max = xAbsDiff;
            /*for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
            {
                max = XMath.Max(Warp.ShuffleDown(max, offset), max);
            }*/
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
                Warp.Barrier();
            }
            Warp.Barrier();
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
            if (warpIdx == 0 & Warp.LaneIdx + 1 < maxShared)
            {
                xDiff = sharedXDiff[Warp.LaneIdx];
                xAbsDiff = sharedXAbsDiff[Warp.LaneIdx];
                xIndex = sharedXIndex[Warp.LaneIdx];
                yIndex = sharedYIndex[Warp.LaneIdx];

                Warp.Barrier();
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
                    Warp.Barrier();
                }
                Warp.Barrier();
                if (Warp.IsFirstLane)
                {
                    xDiffOut[gridIdx, 0] = xDiff;
                    xAbsDiffOut[gridIdx, 0] = xAbsDiff;
                    xIndexOut[gridIdx, 0] = xIndex;
                    yIndexOut[gridIdx, 0] = yIndex;
                }
            }
            /*
            xDiffOut[gridIdx, threadID] = xDiff;
            xAbsDiffOut[gridIdx, threadID] = xAbsDiff;
            xIndexOut[gridIdx, threadID] = xIndex;
            yIndexOut[gridIdx, threadID] = yIndex;*/
        }

        #endregion


        private static void TestSubview(Accelerator accelerator, float[,] xImageIn)
        {
            var maxGroups = accelerator.MaxNumThreads / accelerator.MaxNumThreadsPerGroup;
            var groupThreadIdx = new GroupedIndex(maxGroups, accelerator.MaxNumThreadsPerGroup);
            maxGroups = 2;
            groupThreadIdx = new GroupedIndex(maxGroups, 4);


            var size = new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1));

            var testKernel = accelerator.LoadStreamKernel<GroupedIndex, ArrayView2D<float>>(TestKernel);
            using (var xImage = accelerator.Allocate<float>(size))
            {
                xImage.CopyFrom(xImageIn, new Index2(0, 0), new Index2(0, 0), new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1)));
                int i = 0;
                testKernel(groupThreadIdx, xImage.View);
                accelerator.Synchronize();

                Console.WriteLine("iteration " + i);
            }
        }

        private static void Iteration(Accelerator accelerator, float[,] xImageIn, float[,] bMapIn, float[,] aMapIn, float[,] psf2In, float lambda, float alpha)
        {
            var maxGroups = accelerator.MaxNumThreads / accelerator.MaxNumThreadsPerGroup;
            var groupThreadIdx = new GroupedIndex(maxGroups, accelerator.MaxNumThreadsPerGroup);
            //maxGroups = 2;
            //groupThreadIdx = new GroupedIndex(maxGroups, 4);

            var shrinkReduce = accelerator.LoadStreamKernel<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>>(ShrinkReduceKernel);
            var shrink = accelerator.LoadStreamKernel<GroupedIndex, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<int>, ArrayView2D<int>>(ShrinkKernel);
            var testShuffle = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>, ArrayView<int>>(TestShuffle);

            

            var size = new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1));
            var psfSize = new Index2(psf2In.GetLength(0), psf2In.GetLength(1));

            using (var xImage = accelerator.Allocate<float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))

            using (var maxDiff = accelerator.Allocate<float>(maxGroups))
            using (var maxAbsDiff = accelerator.Allocate<float>(maxGroups))
            using (var xIndex = accelerator.Allocate<int>(maxGroups))
            using (var yIndex = accelerator.Allocate<int>(maxGroups))

            using (var maxDiffShrink = accelerator.Allocate<float>(new Index2(maxGroups, accelerator.MaxNumThreadsPerGroup)))
            using (var maxAbsDiffShrink = accelerator.Allocate<float>(new Index2(maxGroups, accelerator.MaxNumThreadsPerGroup)))
            using (var xIndexShrink = accelerator.Allocate<int>(new Index2(maxGroups, accelerator.MaxNumThreadsPerGroup)))
            using (var yIndexShrink = accelerator.Allocate<int>(new Index2(maxGroups, accelerator.MaxNumThreadsPerGroup)))

            using (var outTest = accelerator.Allocate<int>(32))
            using (var outTest2 = accelerator.Allocate<int>(32))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                xImage.CopyFrom(xImageIn, new Index2(0, 0), new Index2(0, 0), new Index2(xImageIn.GetLength(0), xImageIn.GetLength(1)));
                xCandidates.CopyFrom(bMapIn, new Index2(0, 0), new Index2(0, 0), new Index2(bMapIn.GetLength(0), bMapIn.GetLength(1)));
                aMap.CopyFrom(aMapIn, new Index2(0, 0), new Index2(0, 0), new Index2(aMapIn.GetLength(0), aMapIn.GetLength(1)));
                //psf2.CopyFrom(psf2Input, new Index2(0, 0), new Index2(0, 0), new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1)));

                lambdaAlpha.CopyFrom(lambda, new Index(0));
                lambdaAlpha.CopyFrom(alpha, new Index(1));

                int i = 0;
                shrinkReduce(groupThreadIdx, xImage.View, xCandidates.View, aMap.View, lambdaAlpha.View, maxDiff.View, maxAbsDiff.View, xIndex.View, yIndex.View);
                shrink(groupThreadIdx, xImage.View, xCandidates.View, aMap.View, lambdaAlpha.View, maxDiffShrink.View, maxAbsDiffShrink.View, xIndexShrink.View, yIndexShrink.View);
                testShuffle(new Index(32), outTest.View, outTest2.View);
                accelerator.Synchronize();

                Console.WriteLine("iteration " + i);
                var test = outTest.GetAsArray();
                var test2 = outTest2.GetAsArray();

                var s0 = maxDiffShrink.GetAs2DArray();
                var s1 = maxAbsDiffShrink.GetAs2DArray();
                var s2 = xIndexShrink.GetAs2DArray();
                var s3 = yIndexShrink.GetAs2DArray();

                var maxDiffT = 0.0f;
                var maxAbsDiffT = 0.0f;
                var xIndexT = -1;
                var yIndexT = -1;
                for(int i1 = 0; i1 < s1.GetLength(0); i1++)
                    //for(int j = 0; j < s1.GetLength(1);j++)
                    for (int j = 0; j < 1;j++)
                        if(maxAbsDiffT < s1[i1,j])
                        {
                            maxDiffT = s0[i1, j];
                            maxAbsDiffT = s1[i1, j];
                            xIndexT = s2[i1, j];
                            yIndexT = s3[i1, j];
                        }
                var bla0 = maxDiff.GetAsArray();
                var bla1 = xIndex.GetAsArray();
                var bla2 = yIndex.GetAsArray();
                var x = xImage.GetAsArray();
                var candidate = xCandidates.GetAsArray();
                FitsIO.Write(CopyToImage(x, size), "xImageGPU.fits");
                FitsIO.Write(CopyToImage(candidate, size), "candidateGPU.fits");

            }
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

        public static bool Deconvolve(double[,] xImage, double[,] b, double[,] psf, double lambda, double alpha)
        {
            using (var context = new Context())
            {

                var yPsfHalf = psf.GetLength(0) / 2;
                var xPsfHalf = psf.GetLength(1) / 2;

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
                        var xTest = new float[256, 256];
                        xTest[0, 0] = 1;
                        xTest[0, 1] = 2;
                        xTest[0, 2] = 3;
                        xTest[1, 0] = 20;
                        xTest[2, 0] = 30;

                        xTest[128, 0] = 101;
                        xTest[128, 1] = 102;
                        xTest[129, 0] = 110;

                        xTest[255, 127] = 999;
                        xTest[0, 128] = 1001;
                        xTest[1, 128] = 1010;
                        xTest[0, 129] = 1001;
                        TestSubview(accelerator, xTest);

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
    }
}
