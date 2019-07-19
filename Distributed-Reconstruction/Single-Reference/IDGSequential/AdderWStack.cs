using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace Single_Reference.IDGSequential
{
    class AdderWStack
    {
        public static Complex[,,] AddHack(GriddingConstants c, List<List<SubgridHack>> metadata, List<List<Complex[,]>> subgrids)
        {
            if (c.SubgridsPrecomputed == null)
                throw new InvalidOperationException("AdderWStack should be called when GriddingConstants.WStep > 0");

            var grid = new Complex[c.WLayerCount, c.GridSize, c.GridSize];
            for (int baseline = 0; baseline < subgrids.Count; baseline++)
            {
                var blMeta = metadata[baseline];
                var blSubgridsData = subgrids[baseline];
                for (int subgrid = 0; subgrid < blSubgridsData.Count; subgrid++)
                {
                    var meta = blMeta[subgrid];
                    var data = blSubgridsData[subgrid];

                    int subgridX = meta.UPixel;
                    int subgridY = meta.VPixel;
                    int subgridW = meta.WLambda;

                    // Mirror subgrid coordinates for negative w-values
                    bool negativeW = subgridW < 0;
                    if (negativeW)
                    {
                        subgridX = c.GridSize - subgridX - c.SubgridSize + 1;
                        subgridY = c.GridSize - subgridY - c.SubgridSize + 1;
                        subgridW = -subgridW - 1;
                    }

                    // Check whether subgrid fits in grid
                    if (!(subgridX >= 1 && subgridX < c.GridSize - c.SubgridSize &&
                          subgridY >= 1 && subgridY < c.GridSize - c.SubgridSize)) continue;

                    for (int y = 0; y < c.SubgridSize; y++)
                    {
                        int y_mirrored = c.SubgridSize - 1 - y;
                        int y_ = negativeW ? y_mirrored : y;
                        for (int x = 0; x < c.SubgridSize; x++)
                        {
                            int x_mirrored = c.SubgridSize - 1 - x;
                            int x_ = negativeW ? x_mirrored : x;
                            int xSrc = (x_ + (c.SubgridSize / 2)) % c.SubgridSize;
                            int ySrc = (y_ + (c.SubgridSize / 2)) % c.SubgridSize;

                            int xDst = subgridX + x;
                            int yDst = subgridY + y;

                            var phasor = c.SubgridsPrecomputed[y_, x_];
                            var value = phasor * data[ySrc, xSrc];
                            value = negativeW ? Complex.Conjugate(value) : value;
                            grid[subgridW, yDst, xDst] += value;
                        }
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

                    int subgridX = meta.UPixel;
                    int subgridY = meta.VPixel;
                    int subgridW = meta.WLambda;

                    bool negativeW = subgridW < 0;
                    int wLayer = negativeW ? -subgridW - 1 : subgridW;

                    for (int y = 0; y < c.SubgridSize; y++)
                    {
                        for (int x = 0; x < c.SubgridSize; x++)
                        {
                            int xDst = (x + (c.SubgridSize / 2)) % c.SubgridSize;
                            int yDst = (y + (c.SubgridSize / 2)) % c.SubgridSize;

                            int xSrc = negativeW ? c.GridSize - subgridX - x : subgridX + x;
                            int ySrc = negativeW ? c.GridSize - subgridY - y : subgridY + y;

                            if (subgridX >= 1 && subgridX < c.GridSize - c.SubgridSize &&
                                subgridY >= 1 && subgridY < c.GridSize - c.SubgridSize)
                            {
                                var phasor = c.SubgridsPrecomputed[y, x];

                                var value = grid[y, x];
                                value = negativeW ? Complex.Conjugate(value) : value;
                                subgridData[yDst, xDst] = phasor * value;
                            }
                        }
                    }

                }
            }

            return subgrids;
        }
    }
}
