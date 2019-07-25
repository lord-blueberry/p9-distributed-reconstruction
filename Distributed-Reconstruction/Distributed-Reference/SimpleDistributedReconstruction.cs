using System;
using System.Collections.Generic;
using System.Text;
using MPI;
using System.Diagnostics;
using Single_Reference.IDGSequential;

namespace Distributed_Reference
{
    class SimpleDistributedReconstruction
    {
        public static double[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset data, GriddingConstants c)
        {
            var metadata = Partitioner.CreatePartition(c, data.UVW, data.Frequencies);

            return null;
        }
    }
}
