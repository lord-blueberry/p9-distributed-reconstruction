using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using static System.Math;

namespace Single_Reference.IDGSequential
{
    class Adder
    {
        public static Complex[,] AddHack(GriddingConstants c, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgrids)
        {
            var grid = new Complex[c.GridSize, c.GridSize];

            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgridsData = subgrids[baseline];
                for (int subgrid = 0; subgrid < blSubgridsData.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var subgridData = blSubgridsData[subgrid];

                    int gridX = meta.UPixel;
                    int gridY = meta.VPixel;

                    //TODO: gridX >= 0, even though in plan we check that it is >= 1. 
                    if (gridX >= 0 && gridX < c.GridSize - c.SubgridSize &&
                        gridY >= 0 && gridY < c.GridSize - c.SubgridSize)
                    {
                        for(int y = 0; y < c.SubgridSize; y++)
                        {
                            for(int x = 0; x < c.SubgridSize; x++)
                            {
                                int xSrc = (x + (c.SubgridSize / 2)) % c.SubgridSize;
                                int ySrc = (y + (c.SubgridSize / 2)) % c.SubgridSize;
                                double phase = PI * (x + y - c.SubgridSize) / c.SubgridSize;

                                Complex phasor = new Complex(Cos(phase), Sin(phase));
                                grid[gridY + y, gridX + x] += subgridData[ySrc, xSrc] * phasor;
                            }
                        }

                    }
                    else
                    {
                        throw new Exception("invalid grid");
                    }

                }
            }
            return grid;
        }

        public static List<List<Complex[,]>> SplitHack(GriddingConstants c, List<List<SubgridHack>> metadata, Complex[,] grid)
        {
            var subgrids = new List<List<Complex[,]>>(metadata.Count);

            for (int baseline = 0; baseline < metadata.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgridsData = new List<Complex[,]>(blMeta.Count);
                subgrids.Add(blSubgridsData);

                for (int subgrid = 0; subgrid < blMeta.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var subgridData = new Complex[c.SubgridSize, c.SubgridSize];
                    blSubgridsData.Add(subgridData);

                    int gridX = meta.UPixel;
                    int gridY = meta.VPixel;
                    if (gridX >= 0 && gridX < c.GridSize - c.SubgridSize &&
                        gridY >= 0 && gridY < c.GridSize - c.SubgridSize)
                    {
                        for (int y = 0; y < c.SubgridSize; y++)
                        {
                            for (int x = 0; x < c.SubgridSize; x++)
                            {
                                int xDst = (x + (c.SubgridSize / 2)) % c.SubgridSize;
                                int yDst = (y + (c.SubgridSize / 2)) % c.SubgridSize;
                                double phase = -PI * (x + y - c.SubgridSize) / c.SubgridSize;

                                var phasor = new Complex(Cos(phase), Sin(phase));
                                subgridData[yDst, xDst] = grid[gridY + y, gridX + x] * phasor;
                            }
                        }
                    }
                }
            }

            return subgrids;
        }
    }
}
