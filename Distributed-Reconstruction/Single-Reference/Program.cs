using System;

namespace Single_Reference
{
    class Program
    {
        static void Main()
        {
            var size = 9;
            var bla = Math.Sqrt(size);
            var yPatchCount = (int)Math.Floor(Math.Sqrt(size));
            var xPatchCount = (size / yPatchCount);

            for (int node = 0; node < size; node++)
            {
                CalculateLocalPatchDimensions(node, size, 32, 32);
            }


            //DebugMethods.DebugSimulatedMixed();
            //Deconvolution.NaiveGreedyCD.Run();
            DebugMethods.DebugSimulatedWStack();
            //DebugMethods.MeerKATFull();
        }

        private static void CalculateLocalPatchDimensions(int nodeId, int nodeCount, long ySize, long xSize)
        {
            long yPatchCount = (int)Math.Floor(Math.Sqrt(nodeCount));
            long xPatchCount = (nodeCount / yPatchCount);

            long yIdx = nodeId / xPatchCount;
            long xIdx = nodeId % xPatchCount;

            long yPatchOffset = yIdx * (ySize / yPatchCount);
            long xPatchOffset = xIdx * (xSize / xPatchCount);
            var bla = xSize / xPatchCount;

            var yPatchEnd = yIdx + 1 < yPatchCount ? yPatchOffset + ySize / yPatchCount : ySize;
            var xPatchEnd = xIdx + 1 < xPatchCount ? xPatchOffset + xSize / xPatchCount : xSize;
        }
    }
}
