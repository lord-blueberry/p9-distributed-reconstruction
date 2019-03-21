using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using static System.Math;

namespace Single_Machine.NFFT
{
    class Adder
    {
        public static Complex[,] AddHack(GriddingParams p, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgrids)
        {
            var grid = new Complex[p.GridSize, p.GridSize];

            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgridsData = subgrids[baseline];
                for (int subgrid = 0; subgrid < blSubgridsData.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var data = blSubgridsData[subgrid];

                    int gridX = meta.UPixel;
                    int gridY = meta.VPixel;

                    if (gridX >= 0 && gridX < p.GridSize - p.SubgridSize &&
                        gridY >= 0 && gridY < p.GridSize - p.SubgridSize)
                    {
                        for(int y = 0; y < p.SubgridSize; y++)
                        {
                            for(int x = 0; x < p.SubgridSize; x++)
                            {
                                int xSrc = (x + (p.SubgridSize / 2)) % p.SubgridSize;
                                int ySrc = (y + (p.SubgridSize / 2)) % p.SubgridSize;
                                double phase = PI * (x + y - p.SubgridSize) / p.SubgridSize;
                                Complex phasor = new Complex(Cos(phase), Sin(phase));
                                grid[gridY + y, gridX + x] += data[ySrc, xSrc] * phasor;
                            }
                        }

                    }

                }
            }
            return grid;
        }
    }
}
