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



namespace Distributed_Reference
{
    class DistributedParallelReconstruction
    {
        public const int PSF_CUTFACTOR = 8;
        public static float[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset local, GriddingConstants c, int maxCycle)
        {

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackward = new Stopwatch();
            var watchDeconv = new Stopwatch();

            var metadata = Partitioner.CreatePartition(c, local.UVW, local.Frequencies);

            var imgSection = CalculateLocalImageSection(comm.Rank, comm.Size, c.GridSize, c.GridSize);
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);

            //calculate psf and prepare for correlation in the Fourier space
            var psf = ToFloatImage(CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies));
            var psfCut = PSF.Cut(psf, PSF_CUTFACTOR);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, PSF_CUTFACTOR);
            PaddedConvolver bMapCalculator = null;
            if (comm.Rank ==0)
            {
                var PsfCorrelation = PSF.CalcPaddedFourierCorrelation(psfCut, totalSize);
                bMapCalculator = new PaddedConvolver(PsfCorrelation, new Rectangle(0, 0, psfCut.GetLength(0), psfCut.GetLength(1)));
            }
            
            var residualVis = local.Visibilities;
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {

            }

            return null;
        }

        private static IDeconvolver[] SplitIntoSubpatches(Rectangle patchSize) 
        {
            return null;
        }

    }
}
