using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace Single_Machine.IDG
{
    class IDG
    {
        public static double[,] ToImage(GriddingParams p, List<List<SubgridHack>> metadata, Complex[,,] visibilities, double[,,] uvw, double[] frequencies)
        {
            //Plan

            var subgrids = Gridder.ForwardHack(p, metadata, uvw, null, null, frequencies, null);

            //SubgridFFT

            //Adder
            //FFT

            return null;
        }

        public static Complex[,,] ToVisibilities(GriddingParams p)
        {
            //FFT
            //splitter
            //subgridFFT

            //De-Gridder
            return null;
        }
    }
}
