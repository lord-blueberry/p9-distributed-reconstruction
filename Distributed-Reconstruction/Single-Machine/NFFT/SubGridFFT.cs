using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FFTW.NET;
using System.Numerics;

namespace Single_Machine.NFFT
{
    class SubgridFFT
    {
        public static List<List<Complex[,]>> ForwardHack(GriddingParams p, List<List<Complex[,]>> subgrids)
        {
            var output = new List<List<Complex[,]>>(subgrids.Count);
            for (int baseline= 0; baseline < subgrids.Count; baseline++)
            {
                var blSubgrids = subgrids[baseline];
                var blOutput = new List<Complex[,]>(blSubgrids.Count);
                for (int subgrid = 0; subgrid < blSubgrids.Count; subgrid++)
                {
                    var sub = blSubgrids[subgrid];
                    var outFourier = new Complex[p.SubgridSize, p.SubgridSize];
                    using (var imageSpace = new AlignedArrayComplex(16, p.SubgridSize, p.SubgridSize))
                    using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
                    {
                        //copy
                        for (int i = 0; i < p.SubgridSize; i++)
                        {
                            for (int j = 0; j < p.SubgridSize; j++)
                                imageSpace[i, j] = sub[i, j];
                        }
                        DFT.IFFT(imageSpace, fourierSpace);

                        //NORMALIZE

                        for (int i = 0; i < p.SubgridSize; i++)
                        {
                            for (int j = 0; j < p.SubgridSize; j++)
                                outFourier[i, j] = fourierSpace[i, j];
                        }
                    }
                    blOutput.Add(outFourier);
                }
                output.Add(blOutput);
            }

            return output;
        }
    }
}
