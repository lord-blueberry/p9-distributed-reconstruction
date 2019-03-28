using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using static System.Math;

namespace Single_Machine.NFFT
{
    class Gridder
    {

        #region Grid
        public static List<List<Complex[,]>> ForwardHack(GriddingParams p, List<List<SubgridHack>> metadata, double[,,] uvw, double[,,] vis_real, double[,,] vis_imag, double[] frequencies, float[,] spheroidal)
        {
            var wavenumbers = Math.FrequencyToWavenumber(frequencies);
            var imagesize = p.CellSize * p.GridSize;
            var output = new List<List<Complex[,]>>(metadata.Count);
            for (int baseline = 0; baseline < metadata.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgrids = new List<Complex[,]>(blMeta.Count);
                for (int subgrid = 0; subgrid < blMeta.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var subgridOutput = new Complex[p.SubgridSize, p.SubgridSize];

                    // [+ p.SubgridSize / 2 - p.GridSize / 2] undoes shift from Planner
                    var uOffset = (meta.UPixel + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / imagesize);
                    var vOffset = (meta.VPixel + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / imagesize);
                    var tmpW_lambda = p.WStepLambda * (meta.WLambda + 0.5);
                    var wOffset = 2 * PI * tmpW_lambda;     //discrete w-correction, similar to w-stacking

                    for (int y = 0; y < p.SubgridSize; y++)
                    {
                        for (int x = 0; x < p.SubgridSize; x++)
                        {
                            //real and imaginary part of the pixel. We ignore polarization here
                            var pixel = new Complex();

                            //calculate directional cosines. exp(2*PI*j * (u*l + v*m + w*n))
                            var l = ComputeL(x, p.SubgridSize, imagesize);
                            var m = ComputeL(y, p.SubgridSize, imagesize);
                            var n = ComputeN(l, m);

                            int sampleEnd = meta.timeSampleStart + meta.timeSampleCount;
                            for(int time = meta.timeSampleStart; time < sampleEnd; time++)
                            {
                                var u = uvw[baseline, time, 0];
                                var v = uvw[baseline, time, 1];
                                var w = uvw[baseline, time, 2];
                                double phaseIndex = u * l + v * m + w * n;
                                double phaseOffset = uOffset * l + vOffset * m + wOffset * n;

                                for (int channel = 0; channel < wavenumbers.Length; channel++)
                                {
                                    double phase = phaseOffset - (phaseIndex * wavenumbers[channel]);
                                    var phasor = new Complex(Cos(phase), Sin(phase));
                                    var vis = new Complex(vis_real[baseline, time, channel], vis_imag[baseline, time, channel]);

                                    pixel += vis * phasor;
                                }
                            }

                            //idg A-correction goes here

                            var sph = spheroidal[y, x];
                            int xDest = (x + (p.SubgridSize / 2)) % p.SubgridSize;
                            int yDest = (y + (p.SubgridSize / 2)) % p.SubgridSize;
                            subgridOutput[yDest, xDest] = pixel * sph;
                        }
                    }
                    blSubgrids.Add(subgridOutput);

                }
                output.Add(blSubgrids);
            }

            return output;
        }




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
                    var n = ComputeN(l, m);

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

                    //shift pixels. x=0 and y=0 are the center pixel
                    var sph = spheroidal[x, y];
                    int x_dest = (x + (param.SubgridSize / 2)) % param.SubgridSize;
                    int y_dest = (x + (param.SubgridSize / 2)) % param.SubgridSize;
                    subgridR[x, y] = pixelR * sph;
                    subgridI[x, y] = pixelI * sph;
                }
            }
        }

        private static float ComputeL(int x, int subgridSize, float imageSize)
        {
            //TODO: think about removing 0.5f and replacing imagesize / subgrids with cellsize
            return (x - (subgridSize / 2)) * imageSize / subgridSize;
        }
        private static float ComputeN(float l, float m)
        {
            //evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m))
            //accurately for small values of l and m
            //TODO: rework the c++ version of this snipped into here
            var tmp = (l * l) + (m * m);
            return tmp / ((float)(tmp / 1.0f + Sqrt(1.0 - tmp)));
        }
        #endregion

        #region De-grid
        #endregion

    }
}
