using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using static System.Math;

namespace Single_Reference.IDGSequential
{
    class Gridder
    {

        #region Grid
        public static List<List<Complex[,]>> ForwardHack(GriddingConstants c, List<List<SubgridHack>> metadata, double[,,] uvw, Complex[,,] visibilities, double[] frequencies, float[,] spheroidal)
        {
            var wavenumbers = MathFunctions.FrequencyToWavenumber(frequencies);
            var imagesize = c.CellSize * c.GridSize;
            var output = new List<List<Complex[,]>>(metadata.Count);
            for (int baseline = 0; baseline < metadata.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgrids = new List<Complex[,]>(blMeta.Count);
                for (int subgrid = 0; subgrid < blMeta.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var subgridOutput = new Complex[c.SubgridSize, c.SubgridSize];

                    // [+ p.SubgridSize / 2 - p.GridSize / 2] undoes shift from Planner
                    var uOffset = (meta.UPixel + c.SubgridSize / 2 - c.GridSize / 2) * (2 * PI / imagesize);
                    var vOffset = (meta.VPixel + c.SubgridSize / 2 - c.GridSize / 2) * (2 * PI / imagesize);
                    var tmpW_lambda = c.WStepLambda * (meta.WLambda + 0.5);
                    var wOffset = 2 * PI * tmpW_lambda;     //discrete w-correction, similar to w-stacking

                    for (int y = 0; y < c.SubgridSize; y++)
                    {
                        for (int x = 0; x < c.SubgridSize; x++)
                        {
                            //real and imaginary part of the pixel. We ignore polarization here
                            var pixel = new Complex();

                            //calculate directional cosines. exp(2*PI*j * (u*l + v*m + w*n))
                            var l = ComputeL(x, c.SubgridSize, imagesize);
                            var m = ComputeL(y, c.SubgridSize, imagesize);
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
                                    var vis = visibilities[baseline, time, channel];

                                    pixel += vis * phasor;
                                }
                            }

                            //idg A-correction goes here

                            var sph = spheroidal[y, x];
                            int xDest = (x + (c.SubgridSize / 2)) % c.SubgridSize;
                            int yDest = (y + (c.SubgridSize / 2)) % c.SubgridSize;
                            subgridOutput[yDest, xDest] = pixel * sph;
                        }
                    }
                    blSubgrids.Add(subgridOutput);

                }
                output.Add(blSubgrids);
            }

            return output;
        }
        #endregion

        #region De-grid
        public static Complex[,,] BackwardsHack(GriddingConstants c, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgridData, double[,,] uvw, double[] frequencies, float[,] spheroidal)
        {
            var wavenumbers = MathFunctions.FrequencyToWavenumber(frequencies);
            var imagesize = c.CellSize * c.GridSize;

            var outputVis = new Complex[uvw.GetLength(0), uvw.GetLength(1), wavenumbers.Length];
            for (int baseline = 0; baseline < metadata.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgrids = subgridData[baseline];
                for (int subgrid = 0; subgrid < blMeta.Count; subgrid++)
                {
                    //de-apply a-term correction
                    var meta = blMeta[subgrid];
                    var data = blSubgrids[subgrid];
                    var pixels_copy = new Complex[c.SubgridSize, c.SubgridSize];
                    for (int y = 0; y < c.SubgridSize; y++)
                    {
                        for (int x = 0; x < c.SubgridSize; x++)
                        {
                            var sph = spheroidal[y, x];
                            int xSrc = (x + (c.SubgridSize / 2)) % c.SubgridSize;
                            int ySrc = (y + (c.SubgridSize / 2)) % c.SubgridSize;

                            pixels_copy[y, x] = sph * data[ySrc, xSrc];
                        }
                    }

                    var uOffset = (meta.UPixel + c.SubgridSize / 2 - c.GridSize / 2) * (2 * PI / imagesize);
                    var vOffset = (meta.VPixel + c.SubgridSize / 2 - c.GridSize / 2) * (2 * PI / imagesize);
                    var tmpW_lambda = c.WStepLambda * (meta.WLambda + 0.5);
                    var wOffset = 2 * PI * tmpW_lambda;

                    int sampleEnd = meta.timeSampleStart + meta.timeSampleCount;
                    for (int time = meta.timeSampleStart; time < sampleEnd; time++)
                    {
                        var u = uvw[baseline, time, 0];
                        var v = uvw[baseline, time, 1];
                        var w = uvw[baseline, time, 2];

                        for (int channel = 0; channel < wavenumbers.Length; channel++)
                        {
                            var visibility = new Complex();
                            for (int y = 0; y < c.SubgridSize; y++)
                            {
                                for (int x = 0; x < c.SubgridSize; x++)
                                {
                                    //calculate directional cosines. exp(2*PI*j * (u*l + v*m + w*n))
                                    var l = ComputeL(x, c.SubgridSize, imagesize);
                                    var m = ComputeL(y, c.SubgridSize, imagesize);
                                    var n = ComputeN(l, m);

                                    double phaseIndex = u * l + v * m + w * n;
                                    double phaseOffset = uOffset * l + vOffset * m + wOffset * n;
                                    double phase = (phaseIndex * wavenumbers[channel]) - phaseOffset;
                                    var phasor = new Complex(Cos(phase), Sin(phase));
                                    visibility += pixels_copy[y, x] * phasor;
                                }
                            }

                            double scale = 1.0f / (c.SubgridSize * c.SubgridSize);
                            outputVis[baseline, time, channel] = visibility * scale;
                        }
                    }
                        
                }
            }

            return outputVis;
        }
        #endregion

        private static float ComputeL(int x, int subgridSize, float imageSize)
        {
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
    }
}
