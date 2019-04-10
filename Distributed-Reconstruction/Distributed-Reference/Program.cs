using System;
using MPI;
using System.Diagnostics;
using Single_Reference.IDGSequential;
using Single_Reference.Deconvolution;
using static System.Math;
using System.Numerics;

namespace Distributed_Reference
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var env = new MPI.Environment(ref args, MPI.Threading.Serialized))
            {
                var proc = Process.GetCurrentProcess();
                var name = proc.ProcessName;
                Console.WriteLine(" name: " + name);

                System.Threading.Thread.Sleep(17000);
                Console.WriteLine("Hello World! from rank " + Communicator.world.Rank + " (running on " + MPI.Environment.ProcessorName + ")");

                var comm = Communicator.world;
                if (comm.Rank == 0)
                {
                    comm.Send("Rosie", 1, 0);
                    var receive = comm.Receive<string>(Communicator.anySource, 0);
                    Console.WriteLine("Final Message: " + receive);
                }
                else
                {
                    var receive = comm.Receive<string>(comm.Rank - 1, 0);
                    Console.WriteLine("Worker: " + comm.Rank + " Received Message: " + receive);

                    comm.Send(receive + " " + comm.Rank, (comm.Rank + 1) % comm.Size, 0);
                }

               
                System.Threading.Thread.Sleep(5000);

                /*
                //READ DATA
                var uvw = new double[13, 13, 3];
                var frequencies = new double[8];
                var visibilities = new double[13, 13, 8];

                int gridSize = 256;
                int subgridsize = 16;
                int kernelSize = 4;
                //cell = image / grid
                int max_nr_timesteps = 256;
                double cellSize = 0.5 / 3600.0 * PI / 180.0;
                var c = new GriddingConstants(gridSize, subgridsize, kernelSize, max_nr_timesteps, (float)cellSize, 1, 0.0f);
                var metadata = Partitioner.CreatePartition(c, uvw, frequencies);

                var psfGrid = IDG.GridPSF(c, metadata, uvw, frequencies, visibilities.Length);
                */
                var psfGrid = new Complex[2, 3];

                var psf_total = comm.Reduce<Complex[,]>(psfGrid, SequentialSum, 0);
                double[,] psf = null;
                if(comm.Rank == 0)
                {
                    //psf = FFT.GridIFFT(psfGrid);
                    psf = new double[6, 6];
                    psf[3, 3] = 5.0;
                }
                comm.Broadcast(ref psf, 0);

            }
        }

        public static Complex[,] SequentialSum(Complex[,] a, Complex[,] b)
        {
            for(int y = 0; y < a.GetLength(0); y++)
            {
                for(int x = 0; x < a.GetLength(0); x++)
                {
                    a[y, x] += b[y, x];
                }
            }

            return a;
        }
    }
}
