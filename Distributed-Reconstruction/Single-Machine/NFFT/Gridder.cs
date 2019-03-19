using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Single_Machine.NFFT
{
    class Gridder
    {

        #region Grid
        public static void ForwardSubgrid(GriddingParams param, SubgridData data, float[,] spheroidal)
        {
            float[,] subgridR = new float[param.SubgridSize, param.SubgridSize];
            float[,] subgridI = new float[param.SubgridSize, param.SubgridSize];

            for (int y = 0; y < param.SubgridSize; y++)
            {
                for(int x = 0; x < param.SubgridSize; x++)
                {
                    //real and imaginary part of the pixel. We ignore polarization here
                    var pixelR = 0.0f;
                    var pixelI = 0.0f;

                    //compute l m n
                    var l = ComputeL(x, param.SubgridSize, param.ImageSize);
                    var m = ComputeL(y, param.SubgridSize, param.ImageSize);
                    var n = ComputeN();

                    for(int time = 0; time < data.UVW.Count; time++)
                    {
                        var uvw = data.UVW[time];
                        var phaseIndex = uvw.U * l + uvw.V * m + uvw.W * n;
                        var phaseOffset = data.UOffset * l + data.VOffset * m + data.WOffset * n;
                        for (int channel = 0; channel < data.Wavenumbers.Count; channel++)
                        {
                            var phase = phaseOffset - (phaseIndex * data.Wavenumbers[channel]);
                            var visibility = data.Visibilities[channel, time];

                            var phasorR = (float)Cos(phase);
                            var phasorI = (float)Sin(phase);
                            pixelR += visibility.Real * phasorR;
                            pixelI += visibility.Imag * phasorI;
                        }
                    }

                    //A-Projection would be here, but gets ignored

                    var sph = spheroidal[x, y];
                    subgridR[x, y] = pixelR * sph;
                    subgridI[x, y] = pixelI * sph;
                }
            }
        }

        private static float ComputeL(int x, int subgridSize, float imageSize)
        {
            return (x + 0.5f - (subgridSize / 2)) * imageSize / subgridSize;
        }
        private static float ComputeN()
        {
            return 0.0f;
        }
        #endregion

        #region De-grid
        #endregion

    }
}
