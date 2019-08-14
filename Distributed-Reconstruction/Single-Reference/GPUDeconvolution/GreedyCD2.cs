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

        private struct MaxPixel
        {
            int y;
            int x;
            float xDiff;
            float xAbsDiff; 
        }

        private static void SumKernel(Index index,
            ArrayView2D<float> bMap,
            ArrayView<float> output)
        {
            float sum = 0;
            for(int i = 0; i < bMap.Extent.X; i++)
            {
                sum += bMap[new Index2(index.X, i)];
            }
            output[index] = sum;
        }

        #region kernels
        private static void ShrinkReduceKernel(
            GroupedIndex2 indexGroup,
            ArrayView2D<float> xImage,
            ArrayView2D<float> bMap,
            ArrayView2D<float> aMap,
            ArrayView<float> lambdaAlpha,

            ArrayView<float> xDiffOut,
            ArrayView<float> xAbsDiffOut,
            ArrayView<int> yIndexOut,
            ArrayView<int> xIndexOut)
        {
            var nrBLocks = 32;
            var sharedXDiff = SharedMemory.Allocate<float>(nrBLocks);
            var sharedXAbsDiff = SharedMemory.Allocate<float>(nrBLocks);
            var sharedYIndex = SharedMemory.Allocate<int>(nrBLocks);
            var sharedXIndex = SharedMemory.Allocate<int>(nrBLocks);

            //TODO: WRONG
            int blockIdx = 0;
            var warpIdx = Warp.ComputeWarpIdx(64);

            var index = indexGroup.ComputeGlobalIndex();
            if (index.InBounds(xImage.Extent))
            {
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];
                var xOld = xImage[index];
                var xNew = xOld + bMap[index] / aMap[index];
                xNew = GPUShrinkElasticNet(xNew, lambda, alpha);

                var xDiff = xOld - xNew;
                var xAbsDiff = XMath.Abs(xDiff);
                var yIndex = index.Y;
                var xIndex = index.X;

                //warp reduce
                for(int offset = Warp.WarpSize / 2; offset > 0; offset /=2)
                {
                    var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                    if (oXAbsDiff > xAbsDiff)
                    {
                        xAbsDiff = oXAbsDiff;
                        xDiff = Warp.ShuffleDown(xDiff, offset);
                        yIndex = Warp.ShuffleDown(yIndex, offset);
                        xIndex = Warp.ShuffleDown(xIndex, offset);
                    }
                }

                if(Warp.IsFirstLane)
                {
                    sharedXDiff[warpIdx] = xDiff;
                    sharedXAbsDiff[warpIdx] = xAbsDiff;
                    sharedYIndex[warpIdx] = yIndex;
                    sharedXIndex[warpIdx] = xIndex;
                }
                Group.Barrier();

                //warp 0 reduce 
                if (warpIdx == 0 & Warp.LaneIdx + 1 < sharedXDiff.Length)
                {
                    xDiff = sharedXIndex[Warp.LaneIdx];
                    xAbsDiff = sharedXAbsDiff[Warp.LaneIdx];
                    yIndex = sharedYIndex[Warp.LaneIdx];
                    xIndex = sharedXIndex[Warp.LaneIdx];

                    //warp reduce
                    for (int offset = Warp.WarpSize / 2; offset > 0; offset /= 2)
                    {
                        var oXAbsDiff = Warp.ShuffleDown(xAbsDiff, offset);
                        if (oXAbsDiff > xAbsDiff)
                        {
                            xAbsDiff = oXAbsDiff;
                            xDiff = Warp.ShuffleDown(xDiff, offset);
                            yIndex = Warp.ShuffleDown(yIndex, offset);
                            xIndex = Warp.ShuffleDown(xIndex, offset);
                        }
                    }

                    if (Warp.IsFirstLane)
                    {
                        xDiffOut[blockIdx] = xDiff;
                        xAbsDiffOut[blockIdx] = xAbsDiff;
                        yIndexOut[blockIdx] = yIndex;
                        xIndexOut[blockIdx] = xIndex;
                    }
                }

            }
        }

        private static void ShrinkKernel(Index2 index,
            ArrayView2D<float> xImage,
            ArrayView2D<float> xCandidates,
            ArrayView<float> maxCandidate,
            ArrayView<float> lambdaAlpha)
        {
            if (index.InBounds(xImage.Extent))
            {
                var xOld = xImage[index];
                var xCandidate = xCandidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                Atomic.Max(ref maxCandidate[0], XMath.Abs(xOld - xNew));
            }
        }


        #endregion

        public static void TestRowMajor()
        {
            using (var context = new Context())
            {
                var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
                if (gpuIds.Any())
                {
                    using (var accelerator = new CudaAccelerator(context, gpuIds.First().DeviceId))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                    }
                }
                else
                {
                    using (var accelerator = new CPUAccelerator(context, 4))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        var k = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<float>, ArrayView<float>>(SumKernel);
                        var size = new Index2(8192, 8192);
                        using(var input = accelerator.Allocate<float>(size))
                        using(var output = accelerator.Allocate<float>(8192))
                        {
                            k(new Index(8192), input, output);
                            accelerator.Synchronize();
                            var outSum = output.GetAsArray();
                            var inNotSum = input.GetAs2DArray();
                        }

                    }
                }
            }
        }

        private static void Iteration(Accelerator accelerator, float[,] xImageInput, float[,] candidateInput, float[,] aMapInput, float[,] psf2Input, float lambda, float alpha)
        {
            var shrinkKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>>(ShrinkKernel);

            var size = new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1));
            var psfSize = new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1));

            using (var xImage = accelerator.Allocate<float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            //using (var shrinked = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            //using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                xImage.CopyFrom(xImageInput, new Index2(0, 0), new Index2(0, 0), new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1)));
                xCandidates.CopyFrom(candidateInput, new Index2(0, 0), new Index2(0, 0), new Index2(candidateInput.GetLength(0), candidateInput.GetLength(1)));
                aMap.CopyFrom(aMapInput, new Index2(0, 0), new Index2(0, 0), new Index2(aMapInput.GetLength(0), aMapInput.GetLength(1)));
                //psf2.CopyFrom(psf2Input, new Index2(0, 0), new Index2(0, 0), new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1)));

                maxIndices.CopyFrom(-1, new Index(0));
                maxIndices.CopyFrom(-1, new Index(1));
                maxCandidate.CopyFrom(0, new Index(0));

                lambdaAlpha.CopyFrom(lambda, new Index(0));
                lambdaAlpha.CopyFrom(alpha, new Index(1));

                int i = 0;
                shrinkKernel(size, xImage.View, xCandidates.View, maxCandidate.View, lambdaAlpha.View);
                accelerator.Synchronize();

                Console.WriteLine("iteration " + i);
                
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
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                        b[y, x] = b[y, x] / aMap[y, x];

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
    }
}
