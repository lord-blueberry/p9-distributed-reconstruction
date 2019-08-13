using System;
using System.Collections.Generic;
using System.Text;

using ILGPU;
using ILGPU.Lightning;
using ILGPU.ReductionOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.ShuffleOperations;
using System.Linq;

namespace Single_Reference.GPUDeconvolution
{
    public class GreedyCD
    {
        private static float GPUShrinkElasticNet(float value, float lambda, float alpha) => GPUMath.Max(value - lambda * alpha, 0.0f) / (1 + lambda * (1 - alpha));

        #region kernels
        private static void ShrinkKernel(Index2 index,
                                         ArrayView2D<float> xImage,
                                         ArrayView2D<float> xCandidates,
                                         ArrayView2D<float> shrinked,
                                         ArrayView<float> lambdaAlpha)
        {
            if(index.InBounds(shrinked.Extent))
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

                        //update result
                        xImage[index] = xNew;
                        maxAbsDiff[0] = xImage[index] - xNew;
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
                xCandidates[indexCandidate] += (psf2[index] * maxDiff[0]) / aMap[index];
            }
        }

        private static void ResetIndicesKernel(Index index,
                                               ArrayView<int> maxIndices)
        {
            maxIndices[index] = -1;
        }
        #endregion

        private static void Iteration(Accelerator accelerator)
        {
            var shrinkKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>>(ShrinkKernel);
            var maxIndexKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>, ArrayView<float>>(MaxIndexKernel);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateCandidatesKernel);
            var resetKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>>(ResetIndicesKernel);

            var size = new Index2(32, 32);
            var psfSize = new Index2(16, 16);

            using (var xImage = accelerator.Allocate <float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var shrinked = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
            using (var lambdaAlpha = accelerator.Allocate<float>(2))
            {
                xImage.MemSetToZero();
                xCandidates.MemSetToZero();
                psf2.MemSetToZero();

                xCandidates[new Index2(20, 20)] = 4.0f;
                psf2[new Index2(8, 8)] = 2f;
                psf2[new Index2(7, 7)] = 1f;
                psf2[new Index2(7, 8)] = 1f;
                psf2[new Index2(7, 9)] = 1f;

                maxIndices[0] = -1;
                maxIndices[1] = -1;

                lambdaAlpha[0] = 0.1f;
                lambdaAlpha[1] = 1.0f;

                shrinkKernel(size, xImage.View, xCandidates.View, shrinked.View, lambdaAlpha.View);

                if (accelerator.AcceleratorType == AcceleratorType.CPU)
                    accelerator.Reduce(shrinked.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new AtomicMaxFloat());
                else
                    accelerator.Reduce(shrinked.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new MaxFloat());
                accelerator.Synchronize();

                //up to here is good

                maxIndexKernel(size, xImage.View, xCandidates.View, shrinked.View, maxCandidate.View, maxIndices.View, lambdaAlpha.View);
                accelerator.Synchronize();

                updateCandidatesKernel(psfSize, xCandidates.View, aMap.View, psf2.View, maxCandidate.View, maxIndices.View);
                accelerator.Synchronize();

                resetKernel(new Index(2), maxIndices.View);
                accelerator.Synchronize();

                var x = xImage.GetAsArray();
                var candidate = xCandidates.GetAsArray();
                var p = psf2.GetAsArray();
                var maxI = maxIndices.GetAsArray();
            }
        }
        




        public static void Test()
        {
            using (var context = new Context())
            {
                // Create custom CPU context with a warp size > 1
                using (var accelerator = new CPUAccelerator(context, 4, 4))
                {
                    Console.WriteLine($"Performing operations on {accelerator}");

                    Iteration(accelerator);
                    //Reduce(accelerator);
                    //AtomicReduce(accelerator);
                }
            }
        }
    }
}
