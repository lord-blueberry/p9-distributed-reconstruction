﻿using System;
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
        public const int PSF_CUTFACTOR = 16;
        public static float[,] Reconstruct(Intracommunicator comm, DistributedData.LocalDataset local, GriddingConstants c, int maxCycle, float lambda, float alpha)
        {

            var watchTotal = new Stopwatch();
            var watchForward = new Stopwatch();
            var watchBackward = new Stopwatch();
            var watchDeconv = new Stopwatch();

            var metadata = Partitioner.CreatePartition(c, local.UVW, local.Frequencies);

            var patchSize = CalculateLocalImageSection(comm.Rank, comm.Size, c.GridSize, c.GridSize);
            var totalSize = new Rectangle(0, 0, c.GridSize, c.GridSize);

            //calculate psf and prepare for correlation in the Fourier space
            var psf = ToFloatImage(CalculatePSF(comm, c, metadata, local.UVW, local.Flags, local.Frequencies));
            var psfCut = PSF.Cut(psf, PSF_CUTFACTOR);
            var psfUsed = psfCut;
            var psfSquared = PSF.CalcPSFSquared(psfUsed);
            var maxSidelobe = PSF.CalcMaxSidelobe(psf, PSF_CUTFACTOR);
            PaddedConvolver bMapCalculator = null;
            if (comm.Rank == 0)
            {
                var PsfCorrelation = PSF.CalcPaddedFourierCorrelation(psfUsed, totalSize);
                bMapCalculator = new PaddedConvolver(PsfCorrelation, new Rectangle(0, 0, psfUsed.GetLength(0), psfUsed.GetLength(1)));
            }

            var subPatchDeconv = SplitIntoSubpatches(patchSize);
            var residualVis = local.Visibilities;
            var bMapPatch = new float[32, 32];
            var xImagePatch = new float[1, 1];
            for (int cycle = 0; cycle < maxCycle; cycle++)
            {
                var residuals = new float[1, 1];
                bMapCalculator.ConvolveInPlace(residuals);
                //split bMap

                
                ISubpatchDeconvolver subpatchDeconvolver = new FastGreedyCD(totalSize,patchSize, psfUsed, psfSquared);
                for (int subCycle = 0; subCycle < 10; subCycle++)
                {
                    for (int subPatchIter = 0; subPatchIter < 4; subPatchIter++)
                    {
                        
                        var subPatch = GetSubPatch(subPatchIter);
                        subpatchDeconvolver.Deconvolve(subPatch, xImagePatch, bMapPatch, lambda, alpha, 200);


                        //var globalDiffexchange local diff
                        //subPatchDeconv[subPatchIter].Deconvolve(xImg, residuals, 0.5f, 0.8f, 100);
                        //exchange residuals
                    }
                }
            }

            return null;
        }

        private static ISubpatchDeconvolver SplitIntoSubpatches(Rectangle patchSize) 
        {
            return null;
        }

        private static Rectangle GetSubPatch(int id)
        {
            return new Rectangle(0, 0, 0, 0);
        }

    }
}
