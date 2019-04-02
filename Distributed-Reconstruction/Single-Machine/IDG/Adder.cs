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
        public static Complex[,] AddHack(GriddingConstants constants, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgrids)
        {
            var grid = new Complex[constants.GridSize, constants.GridSize];

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
                    if (gridX >= 0 && gridX < constants.GridSize - constants.SubgridSize &&
                        gridY >= 0 && gridY < constants.GridSize - constants.SubgridSize)
                    {
                        for(int y = 0; y < constants.SubgridSize; y++)
                        {
                            for(int x = 0; x < constants.SubgridSize; x++)
                            {
                                int xSrc = (x + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                int ySrc = (y + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                double phase = PI * (x + y - constants.SubgridSize) / constants.SubgridSize;

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
                        for (int y = 0; y < constants.SubgridSize; y++)
                        {
                            for (int x = 0; x < constants.SubgridSize; x++)
                            {
                                int xSrc = (x + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                int ySrc = (y + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                double phase = PI * (x + y - constants.SubgridSize) / constants.SubgridSize;
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

        public static List<List<Complex[,]>> SplitHack(GriddingConstants constants, List<List<SubgridHack>> metadata, Complex[,] grid)
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
                    var data = new Complex[constants.SubgridSize, constants.SubgridSize];
                    blSubgridsData.Add(data);

                    int gridX = meta.UPixel;
                    int gridY = meta.VPixel;
                    if (gridX >= 0 && gridX < constants.GridSize - constants.SubgridSize &&
                        gridY >= 0 && gridY < constants.GridSize - constants.SubgridSize)
                    {
                        for (int y = 0; y < constants.SubgridSize; y++)
                        {
                            for (int x = 0; x < constants.SubgridSize; x++)
                            {
                                int xDst = (x + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                int yDst = (y + (constants.SubgridSize / 2)) % constants.SubgridSize;
                                double phase = -PI * (x + y - constants.SubgridSize) / constants.SubgridSize;
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
