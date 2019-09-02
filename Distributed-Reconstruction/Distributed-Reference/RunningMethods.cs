using System;
using MPI;
using System.Diagnostics;

using Single_Reference;
using static System.Math;
using System.Numerics;
using System.Collections.Generic;
using System.IO;

using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static Single_Reference.Common;

namespace Distributed_Reference
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
                var data = DistributedData.LoadTinyMeerKAT(comm, folder);
                var totalVisCount = comm.Allreduce(data.VisibilitiesCount, (x, y) => x+y);

                int gridSize = 1024;
                int subgridsize = 16;
                int kernelSize = 4;
                int max_nr_timesteps = 512;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(totalVisCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

                comm.Barrier();
                if (comm.Rank == 0)
                    Console.WriteLine("Done Reading, Starting reconstruction");

                var lambda = 0.8f;
                var alpha = 0.05f;
                var reconstruction = SimpleDistributedReconstruction.Reconstruct(comm, data, c, 5, lambda, alpha, 10000);

                if (comm.Rank == 0)
                    FitsIO.Write(reconstruction, "tinyMeerKATReconstruction.fits");
            }
        }

        public static void RunSubsetTinyMeerKAT(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);

                var comm = Communicator.world;
                var folder = @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\";
                var fullData = DistributedData.LoadSubsetTinyMeerKAT(folder);
                var data = DistributedData.SplitDataAmongNodes(comm, fullData);
                var totalVisibilitiesCount = fullData.VisibilitiesCount;

                int gridSize = 1024;
                int subgridsize = 16;
                int kernelSize = 4;
                int max_nr_timesteps = 512;
                double cellSize = 2.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(totalVisibilitiesCount, gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);

                comm.Barrier();
                if (comm.Rank == 0)
                    Console.WriteLine("Done Reading, Starting reconstruction");

                var reconstruction = SimpleDistributedReconstruction.Reconstruct(comm, data, c, 5, 0.5f, 0.4f, 10000);

                if (comm.Rank == 0)
                    FitsIO.Write(reconstruction, "subsetTinyMeerKATReconstruction.fits");
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

                var reconstruction = SimpleDistributedReconstruction.Reconstruct(comm, data, c, 1, 0.5f, 0.8f, 1000);

                if (comm.Rank == 0)
                    FitsIO.Write(reconstruction, "simulatedReconstruction.fits");
                
            }
        }
    }
}
