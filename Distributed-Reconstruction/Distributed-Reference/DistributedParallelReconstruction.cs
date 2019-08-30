using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Diagnostics;
using MPI;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using Single_Reference;
using static Single_Reference.Common;
using static Distributed_Reference.SimpleDistributedReconstruction;

using Distributed_Reference.DistributedDeconvolution;
namespace Distributed_Reference
{
    class DistributedParallelReconstruction
    {
        public static float[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset local, GriddingConstants c, int maxCycle)
        {
            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackward = new Stopwatch();
            var watchDeconv = new Stopwatch();

            var metadata = Partitioner.CreatePartition(c, local.UVW, local.Frequencies);

            //var imgSection = CalculateLocalImageSection(comm.Rank, comm.Size, c.GridSize, c.GridSize);
            var totalImage = new Rectangle(0, 0, c.GridSize, c.GridSize);

            var psf = CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf);

            var residualVis = local.Visibilities;
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {

            }

            return null;
        }

    }
}
