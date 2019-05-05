using System;

namespace DependencyTest
{
    using ILGPU;
    using ILGPU.Runtime;

    class Program
    {

        static void MyKernel(
            Index index,               // The global thread index (1D in this case)
            ArrayView<int> dataView,   // A view to a chunk of memory (1D in this case)
            int constant)              // A sample uniform constant
        {
            dataView[index] = index + constant;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            // Create main context
            using (var context = new Context())
            {
                // For each available accelerator...
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    // Create default accelerator for the given accelerator id
                    using (var accelerator = Accelerator.Create(context, acceleratorId))
                    {
                        Console.WriteLine($"Performing operations on {accelerator}");

                        // Compiles and loads the implicitly grouped kernel with an automatically determined
                        // group size and an associated default stream.
                        // This function automatically compiles the kernel (or loads the kernel from cache)
                        // and returns a specialized high-performance kernel launcher.
                        // Use LoadAutoGroupedKernel to create a launcher that requires an additional accelerator-stream
                        // parameter. In this case the corresponding call will look like this:
                        // var kernel = accelerator.LoadautoGroupedKernel<Index, ArrayView<int>, int>(MyKernel);
                        // For more detail refer to the ImplicitlyGroupedKernels or ExplicitlyGroupedKernels sample.
                        var kernel = accelerator.LoadAutoGroupedStreamKernel<
                            Index, ArrayView<int>, int>(MyKernel);

                        using (var buffer = accelerator.Allocate<int>(1024))
                        {
                            // Launch buffer.Length many threads and pass a view to buffer
                            // Note that the kernel launch does not involve any boxing
                            kernel(buffer.Length, buffer.View, 42);

                            // Wait for the kernel to finish...
                            accelerator.Synchronize();

                            // Resolve and verify data
                            var data = buffer.GetAsArray();
                            for (int i = 0, e = data.Length; i < e; ++i)
                            {
                                if (data[i] != 42 + i)
                                    Console.WriteLine($"Error at element location {i}: {data[i]} found");
                            }
                        }
                    }
                }
            }
        }
    }
}
