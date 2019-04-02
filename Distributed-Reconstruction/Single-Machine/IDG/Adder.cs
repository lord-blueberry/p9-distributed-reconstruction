using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using static System.Math;

namespace Single_Machine.IDG
{
    class Adder
    {
        public static Complex[,] AddHack(GriddingConstants p, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgrids)
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

                    //TODO: gridX >= 0, even though in plan we check that it is >= 1. 
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

                                //phase = 0;
                                if (y == 8)
                                    Console.Write("");

                                Complex phasor = new Complex(Cos(phase), Sin(phase));
                                var d = data[ySrc, xSrc];
                                var tmp = data[ySrc, xSrc] * phasor;
                                grid[gridY + y, gridX + x] += data[ySrc, xSrc] * phasor;
                            }
                        }

                    }
                    else
                    {
                        //throw new Exception("invalid grid");
                        for (int y = 0; y < p.SubgridSize; y++)
                        {
                            for (int x = 0; x < p.SubgridSize; x++)
                            {
                                int xSrc = (x + (p.SubgridSize / 2)) % p.SubgridSize;
                                int ySrc = (y + (p.SubgridSize / 2)) % p.SubgridSize;
                                double phase = PI * (x + y - p.SubgridSize) / p.SubgridSize;
                                //phase = 0;
                                Complex phasor = new Complex(Cos(phase), Sin(phase));
                                grid[gridY + y, gridX + x] += data[ySrc, xSrc] * phasor;
                            }
                        }
                    }

                }
            }
            return grid;
        }

        public static List<List<Complex[,]>> SplitHack(GriddingConstants p, List<List<SubgridHack>> metadata, Complex[,] grid)
        {
            var subgrids = new List<List<Complex[,]>>(metadata.Count);

            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgridsData = new List<Complex[,]>(blMeta.Count);
                subgrids.Add(blSubgridsData);

                for (int subgrid = 0; subgrid < blSubgridsData.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var data = new Complex[p.SubgridSize, p.SubgridSize];
                    blSubgridsData.Add(data);

                    int gridX = meta.UPixel;
                    int gridY = meta.VPixel;
                    if (gridX >= 0 && gridX < p.GridSize - p.SubgridSize &&
                        gridY >= 0 && gridY < p.GridSize - p.SubgridSize)
                    {
                        for (int y = 0; y < p.SubgridSize; y++)
                        {
                            for (int x = 0; x < p.SubgridSize; x++)
                            {
                                int xDst = (x + (p.SubgridSize / 2)) % p.SubgridSize;
                                int yDst = (y + (p.SubgridSize / 2)) % p.SubgridSize;
                                double phase = -PI * (x + y - p.SubgridSize) / p.SubgridSize;
                                var phasor = new Complex(Cos(phase), Sin(phase));
                                data[yDst, xDst] = grid[gridY + y, gridX + x] * phasor;
                            }
                        }
                    }
                }
            }

            return subgrids;
        }
    }
}
