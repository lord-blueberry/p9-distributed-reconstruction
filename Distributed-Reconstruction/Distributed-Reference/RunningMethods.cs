using System;
using MPI;
using System.Diagnostics;

using static System.Math;
using System.Numerics;
using System.Collections.Generic;
using System.IO;

using Core;
using Core.ImageDomainGridder;
using Core.Deconvolution;
using static Core.Common;

namespace DistributedReconstruction
{
    class RunningMethods
    { 
        public static void RunTest(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);
                System.Threading.Thread.Sleep(17000);

                var comm = Communicator.world;
                int sum = comm.Rank;
                var total = comm.Reduce(sum, (a, b) => a + b, 0);
                if (comm.Rank == 0)
                {
                    Console.WriteLine("testing mpi");
                    Console.WriteLine(total);
                }
            }
        }

        public static void RunTinyMeerKAT(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);

                var comm = Communicator.world;

                var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";
                var data = DistributedData.LoadTinyMeerKAT2(comm.Rank, comm.Size, folder);
                var totalVisCount = comm.Allreduce(data.VisibilitiesCount, (x, y) => x+y);

                int gridSize = 3072;
                int subgridsize = 32;
                int kernelSize = 16;
                int max_nr_timesteps = 1024;
                double cellSize = 1.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(totalVisCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

                comm.Barrier();
                if (comm.Rank == 0)
                    Console.WriteLine("Done Reading, Starting reconstruction");

                var lambda = 0.4f;
                var alpha = 0.1f;
                var reconstruction = MPIMajorCycle.Reconstruct(comm, data, c, 2, lambda, alpha, 10000);

                if (comm.Rank == 0)
                    FitsIO.Write(reconstruction, "tinyMeerKATReconstruction.fits");
            }
        }


        public static void RunSimulated(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);
                //System.Threading.Thread.Sleep(17000);

                var comm = Communicator.world;
                //READ DATA

                var folder = @"C:\dev\GitHub\p9-data\small\fits\simulation_point\";
                var bla = @"C:\dev\GitHub\p9-data\small\fits\simulation_point";
                var fullData = DistributedData.LoadSimulated(folder);
                var data = DistributedData.SplitDataAmongNodes(comm, fullData);
                var totalVisibilitiesCount = fullData.VisibilitiesCount;

                int gridSize = 256;
                int subgridsize = 16;
                int kernelSize = 8;
                int max_nr_timesteps = 512;
                double cellSize = 1.0 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(totalVisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

                comm.Barrier();
                if (comm.Rank == 0)
                    Console.WriteLine("Done Reading, Starting reconstruction");

                var reconstruction = MPIMajorCycle.Reconstruct(comm, data, c, 1, 0.5f, 0.8f, 1000);

                if (comm.Rank == 0)
                    FitsIO.Write(reconstruction, "simulatedReconstruction.fits");
                
            }
        }
    }
}
