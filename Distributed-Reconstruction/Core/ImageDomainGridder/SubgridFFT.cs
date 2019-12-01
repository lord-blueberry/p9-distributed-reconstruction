using System;
using System.Collections.Generic;
using System.Text;
using FFTW.NET;
using System.Numerics;

namespace Core.ImageDomainGridder
{
    class SubgridFFT
    {
        public static List<List<Complex[,]>> Forward(GriddingConstants c, List<List<Complex[,]>> subgrids)
        {
            var output = new List<List<Complex[,]>>(subgrids.Count);
            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blSubgrids = subgrids[baseline];
                var blOutput = new List<Complex[,]>(blSubgrids.Count);
                for (int subgrid = 0; subgrid < blSubgrids.Count; subgrid++)
                {
                    var sub = blSubgrids[subgrid];
                    var outFourier = new Complex[c.SubgridSize, c.SubgridSize];
                    using (var imageSpace = new AlignedArrayComplex(16, c.SubgridSize, c.SubgridSize))
                    using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
                    {
                        //copy
                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                imageSpace[i, j] = sub[i, j];
                        }

                        /*
                         * This is not a bug
                         * The original IDG implementation uses the inverse Fourier transform here, even though the
                         * Subgrids are already in image space.
                         */
                        DFT.IFFT(imageSpace, fourierSpace);
                        var norm = 1.0 / (c.SubgridSize * c.SubgridSize);

                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                outFourier[i, j] = fourierSpace[i, j] * norm;
                        }
                    }
                    blOutput.Add(outFourier);
                }
                output.Add(blOutput);
            }

            return output;
        }

        public static List<List<Complex[,]>> Backward(GriddingConstants c, List<List<Complex[,]>> subgrids)
        {
            var output = new List<List<Complex[,]>>(subgrids.Count);
            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blSubgrids = subgrids[baseline];
                var blOutput = new List<Complex[,]>(blSubgrids.Count);
                for (int subgrid = 0; subgrid < blSubgrids.Count; subgrid++)
                {
                    var sub = blSubgrids[subgrid];
                    var outFourier = new Complex[c.SubgridSize, c.SubgridSize];
                    using (var imageSpace = new AlignedArrayComplex(16, c.SubgridSize, c.SubgridSize))
                    using (var fourierSpace = new AlignedArrayComplex(16, imageSpace.GetSize()))
                    {
                        //copy
                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
                                imageSpace[i, j] = sub[i, j];
                        }

                        DFT.FFT(imageSpace, fourierSpace);
                        //normalization is done in the Gridder

                        for (int i = 0; i < c.SubgridSize; i++)
                        {
                            for (int j = 0; j < c.SubgridSize; j++)
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
