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
    public class GreedyCD
    {
        
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => XMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region kernels
        private static void ShrinkKernel(Index2 index,
            ArrayView2D<float> xImage,
            ArrayView2D<float> xCandidates,
            ArrayView<float> maxCandidate,
            ArrayView<float> lambdaAlpha)
        {
            if(index.InBounds(xImage.Extent))
            {
                var xOld = xImage[index];
                var xCandidate = xCandidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                Atomic.Max(ref maxCandidate[0], XMath.Abs(xOld - xNew));
            }
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
                var update = (psf2[index] * maxDiff[0]) / aMap[indexCandidate];
                xCandidates[indexCandidate] += (psf2[index] * maxDiff[0]) / aMap[indexCandidate];
            }
        }

        private static void ResetIndicesKernel(Index index,
            ArrayView<int> maxIndices,
            ArrayView<float> maxCandidate)
        {
            maxIndices[index] = -1;
            maxCandidate[0] = 0;
        }
        #endregion

        private static void Iteration(Accelerator accelerator, float[,] xImageInput, float[,]candidateInput, float[,] aMapInput, float[,] psf2Input, float lambda, float aplpha)
        {
            var shrinkKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<float>>(ShrinkKernel);
            var maxIndexKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>>(Shrink2);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateCandidatesKernel);
            var resetKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>, ArrayView<float>>(ResetIndicesKernel);

            var size = new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1));
            var psfSize = new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1));

            using (var xImage = accelerator.Allocate <float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var shrinked = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                xImage.CopyFrom(xImageInput, new Index2(0, 0), new Index2(0, 0), new Index2(0, 0));
                xCandidates.CopyFrom(candidateInput, new Index2(0, 0), new Index2(0, 0), new Index2(0, 0));
                aMap.CopyFrom(aMapInput, new Index2(0, 0), new Index2(0, 0), new Index2(0, 0));
                psf2.CopyFrom(psf2Input, new Index2(0, 0), new Index2(0, 0), new Index2(0, 0));

                maxIndices.CopyFrom(-1, new Index(0));
                maxIndices.CopyFrom(-1, new Index(1));
                maxCandidate.CopyFrom(0, new Index(0));

                lambdaAlpha.View[0] = lambda;
                lambdaAlpha.View[1] = aplpha;

                for(int i = 0; i< 100; i++)
                {
                    shrinkKernel(size, xImage.View, xCandidates.View, maxCandidate.View, lambdaAlpha.View);
                    accelerator.Synchronize();

                    maxIndexKernel(size, xImage.View, xCandidates.View, maxCandidate.View, maxIndices.View, lambdaAlpha.View);
                    accelerator.Synchronize();
                    var indices = maxIndices.GetAsArray();
                    var maxDiff = maxCandidate.GetAsArray();

                    updateCandidatesKernel(psfSize, xCandidates.View, aMap.View, psf2.View, maxCandidate.View, maxIndices.View);
                    accelerator.Synchronize();

                    resetKernel(new Index(2), maxIndices.View, maxCandidate.View);
                    accelerator.Synchronize();
                    Console.WriteLine("iteration " + i);
                }


                var x = xImage.GetAsArray();
                var candidate = xCandidates.GetAsArray();
                var p = psf2.GetAsArray();
                FitsIO.Write(CopyToImage(x, size), "xImageGPU.fits");
                FitsIO.Write(CopyToImage(candidate, size), "candidateGPU.fits");
                
            }
        }

        private static double[,] CopyToImage(float[] img, Index2 size)
        {
            var output = new double[size.Y, size.X];
            var index = 0;
            for(int y = 0; y < size.Y; y++)
            {
                for (int x = 0; x < size.X; x++)
                    output[y, x] = img[index + x];
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

                var gpuId = Accelerator.Accelerators.Where(id => id.AcceleratorType == AcceleratorType.Cuda).First();
                if(gpuId != null)
                {
                    using (var accelerator = new CudaAccelerator(context, gpuId.DeviceId))
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
    
        public static void Test()
        {
            using (var context = new Context())
            {
                // Create custom CPU context with a warp size > 1
                using (var accelerator = new CPUAccelerator(context, 4))
                {
                    Console.WriteLine($"Performing operations on {accelerator}");

                    //Iteration(accelerator);
                    //Reduce(accelerator);
                    //AtomicReduce(accelerator);
                }
            }
        }

        private static float[,] ToFloat(double[,] img)
        {
            var output = new float[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                    output[i, j] = (float)img[i, j];
            return output;
        }





        #region V0.3.0 code
        /*
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => GPUMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region kernels
        private static void ShrinkKernel(Index2 index,
                                         ArrayView2D<float> xImage,
                                         ArrayView2D<float> xCandidates,
                                         ArrayView2D<float> shrinked,
                                         ArrayView<float> lambdaAlpha)
        {
            if (index.InBounds(shrinked.Extent))
            {
                var xOld = xImage[index];
                var xCandidate = xCandidates[index];
                var lambda = lambdaAlpha[0];
                var alpha = lambdaAlpha[1];

                var xNew = GPUShrinkElasticNet(xOld + xCandidate, lambda, alpha);
                shrinked[index] = GPUMath.Abs(xOld - xNew);
            }
        }

        private static void MaxIndexKernel(Index2 index,
                                       ArrayView2D<float> xImage,
                                       ArrayView2D<float> xCandidate,
                                       ArrayView2D<float> shrinked,
                                       ArrayView<float> maxAbsDiff,
                                       ArrayView<int> maxIndices,
                                       ArrayView<float> lambdaAlpha)
        {
            //not sure if necessary, but bounds check were always done in the ILGPU examples
            if (index.InBounds(xImage.Extent))
            {
                //TODO: fix this line for ximage.size != xCandidates.size
                var shrink = shrinked[index];
                var max = maxAbsDiff[0];

                if (shrink == max)
                {
                    var oldValue = Atomic.CompareExchange(maxIndices.GetVariableView(0), -1, index.Y);
                    if (oldValue == -1)
                    {
                        maxIndices[1] = index.X;

                        //retrieve sign of maximum candidate
                        var lambda = lambdaAlpha[0];
                        var alpha = lambdaAlpha[1];
                        var xNew = GPUShrinkElasticNet(xImage[index] + xCandidate[index], lambda, alpha);
                        maxAbsDiff[0] = xImage[index] - xNew;

                        //update result
                        xImage[index] = xNew;
                    }
                }
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
                xCandidates[indexCandidate] += (psf2[index] * maxDiff[0]) / aMap[indexCandidate];
            }
        }

        private static void ResetIndicesKernel(Index index,
                                               ArrayView<int> maxIndices)
        {
            maxIndices[index] = -1;
        }
        #endregion

        private static void Iteration(Accelerator accelerator, double[,] xImageInput, double[,] candidateInput, double[,] aMapInput, double[,] psf2Input, float lambda, float aplpha)
        {
            var shrinkKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>(ShrinkKernel);
            var maxIndexKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>>(MaxIndexKernel);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateCandidatesKernel);
            var resetKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>>(ResetIndicesKernel);

            var size = new Index2(xImageInput.GetLength(0), xImageInput.GetLength(1));
            var psfSize = new Index2(psf2Input.GetLength(0), psf2Input.GetLength(1));

            using (var xImage = accelerator.Allocate<float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var shrinked = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                CopyToBuffer(xImage, xImageInput);
                CopyToBuffer(xCandidates, candidateInput);
                CopyToBuffer(aMap, aMapInput);
                CopyToBuffer(psf2, psf2Input);

                maxIndices[0] = -1;
                maxIndices[1] = -1;

                lambdaAlpha[0] = lambda;
                lambdaAlpha[1] = aplpha;

                for(int i = 0; i < 100; i++)
                {
                    shrinkKernel(size, xImage.View, xCandidates.View, shrinked.View, lambdaAlpha.View);
                    accelerator.Synchronize();

                    if (accelerator.AcceleratorType == AcceleratorType.CPU)
                        accelerator.Reduce(shrinked.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new AtomicMaxFloat());
                    else
                        accelerator.Reduce(shrinked.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new MaxFloat());
                    accelerator.Synchronize();

                    maxIndexKernel(size, xImage.View, xCandidates.View, shrinked.View, maxCandidate.View, maxIndices.View, lambdaAlpha.View);
                    accelerator.Synchronize();
                    //var indices = maxIndices.GetAsArray();
                    //var maxDiff = maxCandidate.GetAsArray();

                    updateCandidatesKernel(psfSize, xCandidates.View, aMap.View, psf2.View, maxCandidate.View, maxIndices.View);
                    accelerator.Synchronize();

                    resetKernel(new Index(2), maxIndices.View);
                    accelerator.Synchronize();
                    Console.WriteLine("iteration " + i);
                }

                var x = xImage.GetAsArray();
                var candidate = xCandidates.GetAsArray();
                var p = psf2.GetAsArray();
                FitsIO.Write(CopyToImage(x, size), "xImageGPU.fits");
                FitsIO.Write(CopyToImage(candidate, size), "candidateGPU.fits");

            }
        }

        private static void CopyToBuffer(MemoryBuffer2D<float> buffer, double[,] image)
        {
            for (int i = 0; i < image.GetLength(0); i++)
                for (int j = 0; j < image.GetLength(1); j++)
                    buffer[new Index2(j, i)] = (float)image[i, j];
        }

        private static double[,] CopyToImage(float[] img, Index2 size)
        {
            var output = new double[size.Y, size.X];
            var index = 0;
            for (int y = 0; y < size.Y; y++)
            {
                for (int x = 0; x < size.X; x++)
                    output[y, x] = img[index + x];
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

                var gpuId = Accelerator.Accelerators.Where(id => id.AcceleratorType == AcceleratorType.Cuda);
                if (gpuId != null)
                {
                    using (var accelerator = new CudaAccelerator(context))
                    {
                        Iteration(accelerator, xImage, b, aMap, psf2, (float)lambda, (float)alpha);
                    }
                }
                else
                {
                    using (var accelerator = new CPUAccelerator(context, 4, 4))
                    {
                        Iteration(accelerator, xImage, b, aMap, psf2, (float)lambda, (float)alpha);
                    }
                }
            }

            return true;
        }

        public static void Test()
        {
            using (var context = new Context())
            {
                // Create custom CPU context with a warp size > 1

                using (var accelerator = new CPUAccelerator(context, 4, 4))
                {
                    Console.WriteLine($"Performing operations on {accelerator}");

                    //Iteration(accelerator);
                    //Reduce(accelerator);
                    //AtomicReduce(accelerator);
                }
            }
        }
    */
        #endregion




    }
}
