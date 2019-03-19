using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using FFTW.NET;

namespace Single_Machine
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            using (var input = new AlignedArrayComplex(16, 64, 24))
            using (var output = new AlignedArrayComplex(16, input.GetSize()))
            {
                for (int row = 0; row < input.GetLength(0); row++)
                {
                    for (int col = 0; col < input.GetLength(1); col++)
                        input[row, col] = (double)row * col / input.Length;
                }

                DFT.FFT(input, output);
                DFT.IFFT(output, output);

                for (int row = 0; row < input.GetLength(0); row++)
                {
                    for (int col = 0; col < input.GetLength(1); col++)
                        Console.Write((output[row, col].Real / input[row, col].Real / input.Length).ToString("F2").PadLeft(6));
                    Console.WriteLine();
                }
            }
            */


            using (var timeDomain = new AlignedArrayComplex(16, 64, 24))
            using (var frequencyDomain = new AlignedArrayComplex(16, timeDomain.GetSize()))
            using (var fft = FftwPlanC2C.Create(timeDomain, frequencyDomain, DftDirection.Forwards))
            using (var ifft = FftwPlanC2C.Create(frequencyDomain, timeDomain, DftDirection.Backwards))
            {
                // Set the input after the plan was created as the input may be overwritten
                // during planning
                for (int row = 0; row < timeDomain.GetLength(0); row++)
                {
                    for (int col = 0; col < timeDomain.GetLength(1); col++)
                        timeDomain[row, col] = (double)row * col / timeDomain.Length;
                }

                // timeDomain -> frequencyDomain
                fft.Execute();

                for (int i = frequencyDomain.Length / 2; i < frequencyDomain.Length; i++)
                    frequencyDomain[i] = 0;

                // frequencyDomain -> timeDomain
                ifft.Execute();

                // Do as many forwards and backwards transformations here as you like
            }
        }



    }
}
