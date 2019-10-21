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
using static Single_Reference.GPUDeconvolution.GreedyCD2;

namespace Single_Reference.GPUDeconvolution
{
    class StupidGreedy
    {
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => XMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region kernels
        private static void ShrinkKernel(Index2 index,
            ArrayView2D<float> xImage,
            ArrayView2D<float> xCandidates,
            ArrayView<float> maxCandidate,
            ArrayView<float> lambdaAlpha,
            ArrayView<Pixel> output)
        {
            if(index.X == 0 & index.Y == 0)
                output[0].AbsDiff = 0;
            
            if (index.InBounds(xImage.Extent))
            {
                var xOld = xImage[index];
                var xCandidate = xCandidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                //Atomic.Max(ref maxCandidate[0], XMath.Abs(xOld - xNew));
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
                Atomic.MakeAtomic(ref output[0], pix, new MaxPixelOperation(), new MaxPixelCompareExchange());
            }
        }

        private static void UpdateX (
            ILGPU.Index index,
            ArrayView2D<float> xImage,
            ArrayView<Pixel> pixel)
        {
            xImage[pixel[0].X, pixel[0].Y] += pixel[0].Sign * pixel[0].AbsDiff;
        }

        private static void Shrink2(Index2 index,
            ArrayView2D<float> xImage,
            ArrayView2D<float> xCandidates,
            ArrayView<float> maxCandidate,
            ArrayView<int> maxIndices,
            ArrayView<float> lambdaAlpha)
        {
            //not sure if necessary, but bounds check were always done in the ILGPU examples
            if (index.InBounds(xImage.Extent))
            {
                //TODO: fix this line for ximage.size != xCandidates.size
                var xOld = xImage[index];
                var xCandidate = xCandidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                var xDiff = xOld - xNew;

                if (maxCandidate[0] == XMath.Abs(xDiff))
                {
                    var oldValue = Atomic.CompareExchange(ref maxIndices[0], -1, index.Y);
                    if (oldValue == -1)
                    {
                        maxIndices[1] = index.X;

                        //retrieve sign of maximum candidat
                        maxCandidate[0] = xDiff;

                        //update result
                        xImage[index] = xNew;
                    }
                }
            }
        }

        private static void UpdateCandidatesKernel2(Index2 index,
            ArrayView2D<float> xCandidates,
            ArrayView2D<float> aMap,
            ArrayView2D<float> psf2,
            ArrayView<Pixel> pixel)
        {
            var indexCandidate = index.Add(new Index2(pixel[0].X, pixel[0].Y)).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(xCandidates.Extent))
            {
                xCandidates[indexCandidate] -= (psf2[index] * pixel[0].Sign * pixel[0].AbsDiff) / aMap[indexCandidate];
            }
        }

        private static void UpdateCandidatesKernel(Index2 index,
            ArrayView2D<float> xCandidates,
            ArrayView2D<float> aMap,
            ArrayView2D<float> psf2,
            ArrayView<float> maxDiff,
            ArrayView<int> maxIndices)
        {
            var indexCandidate = index.Add(new Index2(maxIndices[1], maxIndices[0])).Subtract(psf2.Extent / 2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(xCandidates.Extent))
            {
                //var update = (psf2[index] * maxDiff[0]) / aMap[indexCandidate];
                xCandidates[indexCandidate] += (psf2[index] * maxDiff[0]) / aMap[indexCandidate];
            }
        }

        private static void ResetIndicesKernel(ILGPU.Index index,
            ArrayView<int> maxIndices,
            ArrayView<float> maxCandidate,
            ArrayView<Pixel> maxPixel)
        {
            maxIndices[index] = -1;
            maxCandidate[0] = 0;
            maxPixel[0].AbsDiff = 0;
            maxPixel[0].X = -1;
            maxPixel[0].Y = -1;
        }
        #endregion

        private static void Iteration(Accelerator accelerator, float[,] xImageInput, float[,] candidateInput, float[,] aMapInput, float[,] psf2Input, float lambda, float alpha)
        {
            var shrinkKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>, ArrayView<Pixel>>(ShrinkKernel);
            var maxIndexKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>>(Shrink2);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateCandidatesKernel);
            var resetKernel = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<int>, ArrayView<float>, ArrayView<Pixel>>(ResetIndicesKernel);

            var updateXKernel = accelerator.LoadStreamKernel<ILGPU.Index, ArrayView2D<float>, ArrayView<Pixel>>(UpdateX);
            var updateCandidatesKernel2 = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<Pixel>>(UpdateCandidatesKernel2);

            var size = new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1));
            var psfSize = new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1));

            using (var xImage = accelerator.Allocate<float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var shrinked = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxPixel = accelerator.Allocate<Pixel>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                xImage.CopyFrom(xImageInput, new Index2(0, 0), new Index2(0, 0), new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1)));
                xCandidates.CopyFrom(candidateInput, new Index2(0, 0), new Index2(0, 0), new Index2(candidateInput.GetLength(0), candidateInput.GetLength(1)));
                aMap.CopyFrom(aMapInput, new Index2(0, 0), new Index2(0, 0), new Index2(aMapInput.GetLength(0), aMapInput.GetLength(1)));
                psf2.CopyFrom(psf2Input, new Index2(0, 0), new Index2(0, 0), new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1)));

                maxIndices.CopyFrom(-1, new ILGPU.Index(0));
                maxIndices.CopyFrom(-1, new ILGPU.Index(1));
                maxCandidate.CopyFrom(0, new ILGPU.Index(0));

                lambdaAlpha.CopyFrom(lambda, new ILGPU.Index(0));
                lambdaAlpha.CopyFrom(alpha, new ILGPU.Index(1));

                Console.WriteLine("Start");
                var watch = new System.Diagnostics.Stopwatch();
                watch.Start();
                for (int i = 0; i < 1000; i++)
                {
                    shrinkKernel(size, xImage.View, xCandidates.View, maxCandidate.View, lambdaAlpha.View, maxPixel.View);
                    accelerator.Synchronize();

                    //maxIndexKernel(size, xImage.View, xCandidates.View, maxCandidate.View, maxIndices.View, lambdaAlpha.View);
                    //accelerator.Synchronize();
                    //var indices = maxIndices.GetAsArray();
                    //var maxDiff = maxCandidate.GetAsArray();

                    updateXKernel(new ILGPU.Index(1), xImage.View, maxPixel.View);
                    updateCandidatesKernel2(psfSize, xCandidates.View, aMap.View, psf2.View, maxPixel.View);
                    //updateCandidatesKernel(psfSize, xCandidates.View, aMap.View, psf2.View, maxCandidate.View, maxIndices.View);
                    accelerator.Synchronize();

                    resetKernel(new ILGPU.Index(2), maxIndices.View, maxCandidate.View, maxPixel.View);
                    accelerator.Synchronize();
                    //Console.WriteLine("iteration " + i);
                }
                watch.Stop();
                Console.WriteLine(watch.Elapsed);
                var x = xImage.GetAs2DArray();
                var candidate = xCandidates.GetAs2DArray();
                var p = psf2.GetAsArray();
                FitsIO.Write(ToDouble(x), "xImageGPU.fits");
                FitsIO.Write(ToDouble(candidate), "candidateGPU.fits");
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

                var psf2 = Common.PSF.CalcPSFSquared(Common.ToFloatImage(psf));
                var aMap = CommonDeprecated.PSF.CalcAMap(xImage, psf);
                for (int y = 0; y < b.GetLength(0); y++)
                    for (int x = 0; x < b.GetLength(1); x++)
                        b[y, x] = b[y, x] / aMap[y, x];

                var gpuIds = Accelerator.Accelerators.Where(id => id.AcceleratorType != AcceleratorType.CPU);
                if (gpuIds.Any())
                {
                    using (var accelerator = new CudaAccelerator(context, gpuIds.First().DeviceId))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        Iteration(accelerator, ToFloat(xImage), ToFloat(b), ToFloat(aMap), psf2, (float)lambda, (float)alpha);
                    }
                }
                else
                {
                    using (var accelerator = new CPUAccelerator(context, 4))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");
                        Iteration(accelerator, ToFloat(xImage), ToFloat(b), ToFloat(aMap), psf2, (float)lambda, (float)alpha);
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

        private static double[,] ToDouble(float[,] img)
        {
            var output = new double[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                    output[i, j] = img[i, j];
            return output;
        }


    }
}
