using System;
using System.Collections.Generic;
using System.Text;

using ILGPU;
using ILGPU.Lightning;
using ILGPU.Lightning.Sequencers;
using ILGPU.ReductionOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.ShuffleOperations;
using System.Linq;

namespace Single_Reference.GPUDeconvolution
{
    public class GreedyCD
    {
        private static void Iteration(Accelerator accelerator)
        {
            var maxIndexKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(MaxIndexKernel);
            var updateCandidatesKernel = accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<float>, ArrayView2D<float>, ArrayView2D<float>, ArrayView<float>, ArrayView<int>>(UpdateCandidatesKernel);
            var resetKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>>(ResetIndicesKernel);

            var size = new Index2(32, 32);
            var psfSize = new Index2(16, 16);

            using (var xImage = accelerator.Allocate <float>(size))
            using (var xCandidates = accelerator.Allocate<float>(size))
            using (var aMap = accelerator.Allocate<float>(size))
            using (var psf2 = accelerator.Allocate<float>(psfSize))
            using (var maxCandidate = accelerator.Allocate<float>(1))
            using (var maxIndices = accelerator.Allocate<int>(2))
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

                if (accelerator.AcceleratorType == AcceleratorType.CPU)
                    accelerator.Reduce(xCandidates.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new AtomicMaxFloat());
                else
                    accelerator.Reduce(xCandidates.View.AsLinearView(), maxCandidate.View, new ShuffleDownFloat(), new MaxFloat());
                accelerator.Synchronize();

                maxIndexKernel(size, xImage.View, xCandidates.View, maxCandidate.View, maxIndices.View);
                accelerator.Synchronize();

                updateCandidatesKernel(psfSize, xCandidates.View, aMap.View, psf2.View, maxCandidate.View, maxIndices.View);
                accelerator.Synchronize();

                resetKernel(new Index(2), maxIndices.View);
                accelerator.Synchronize();

                var x = xImage.GetAsArray();
                var candidate = xCandidates.GetAsArray();
                var maxI = maxIndices.GetAsArray();
            }
        }
        
        private static void MaxIndexKernel(Index2 index,
                                           ArrayView2D<float> xImage,
                                           ArrayView2D<float> xCandidates,
                                           ArrayView<float> maxCandidate,
                                           ArrayView<int> maxIndices)
        {
            //not sure if necessary, but bounds check were always done in the ILGPU examples
            if(index.InBounds(xImage.Extent))
            {
                //TODO: fix this line for ximage.size != xCandidates.size
                var myCandidate = xCandidates[index];
                var max = maxCandidate[0];

                if(myCandidate == max)
                {
                    var oldValue = Atomic.CompareExchange(maxIndices.GetVariableView(0), -1, index.Y);
                    if (oldValue == -1)
                    {
                        maxIndices[1] = index.X;

                        //update result
                        xImage[index] -= max;
                    }   
                }
            }
        }


        private static void UpdateCandidatesKernel(Index2 index,
                                                   ArrayView2D<float> xCandidates,
                                                   ArrayView2D<float> aMap,
                                                   ArrayView2D<float> psf2,
                                                   ArrayView<float> maxCandidate,
                                                   ArrayView<int> maxIndices)
        {
            var indexCandidate = index.Add(new Index2(maxIndices[1], maxIndices[0])).Subtract(psf2.Extent /2);
            if (index.InBounds(psf2.Extent) & indexCandidate.InBounds(xCandidates.Extent))
            {
                xCandidates[indexCandidate] -= psf2[index];
            }
        }

        private static void ResetIndicesKernel(Index index,
                                               ArrayView<int> maxIndices)
        {
            maxIndices[index] = -1;
        }


        /// <summary>
        /// Demonstrates the reduce functionality.
        /// </summary>
        /// <param name="accl">The target accelerator.</param>
        static void Reduce(Accelerator accl)
        {
            var index = new Index2(32, 32);
            using (var buffer = accl.Allocate<int>(index))
            {
                for (int i = 0; i < 32; i++)
                    for (int j = 0; j < 32; j++)
                        buffer[new Index2(i, j)] = i + j;

                using (var target = accl.Allocate<int>(1))
                {
                    
                    // This overload requires an explicit output buffer but
                    // uses an implicit temporary cache from the associated accelerator.
                    // Call a different overload to use a user-defined memory cache.
                    accl.Reduce(
                        buffer.View.AsLinearView(),
                        target.View,
                        new ShuffleDownInt32(),
                        new MaxInt32());

                    accl.Synchronize();

                    var data = target.GetAsArray();
                    for (int i = 0, e = data.Length; i < e; ++i)
                        Console.WriteLine($"Reduced[{i}] = {data[i]}");
                }
            }
        }

        /// <summary>
        /// Demonstrates the reduce functionality.
        /// </summary>
        /// <param name="accl">The target accelerator.</param>
        static void AtomicReduce(Accelerator accl)
        {
            var index = new Index2(32, 32);
            using (var buffer = accl.Allocate<int>(index))
            {

                for (int i = 0; i < 32; i++)
                    for (int j = 0; j < 32; j++)
                        buffer[new Index2(i, j)] = i + j;
                using (var target = accl.Allocate<int>(1))
                {

                    // This overload requires an explicit output buffer but
                    // uses an implicit temporary cache from the associated accelerator.
                    // Call a different overload to use a user-defined memory cache.
                    accl.AtomicReduce(
                        buffer.View.AsLinearView(),
                        target.View,
                        new ShuffleDownInt32(),
                        new AtomicMaxInt32());

                    accl.Synchronize();

                    var data = target.GetAsArray();
                    for (int i = 0, e = data.Length; i < e; ++i)
                        Console.WriteLine($"AtomicReduced[{i}] = {data[i]}");
                }
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
