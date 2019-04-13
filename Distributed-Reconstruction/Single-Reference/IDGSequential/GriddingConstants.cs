using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Single_Reference.IDGSequential
{
    /// <summary>
    /// Immutable class holding the gridding parameters
    /// </summary>
    public class GriddingConstants
    {
        
        public int GridSize { get; }
        public int SubgridSize { get; }
        public int KernelSize { get; }
        
        public int MaxTimestepsPerSubgrid { get; }

        public float ScaleArcSec { get; }
        public float ImageSize { get; }

        public int WLayerCount { get; }
        public float WStep { get; }
        public float WStepLambda { get; }

        public float[,] SubgridSpheroidal { get; }
        public float[,] GridSpheroidal { get; }


        public GriddingConstants(int gridSize, int subgridSize, int kernelSize, int maxTimesteps, float scale, int wLayerCount, float wStep)
        {
            this.GridSize = gridSize;
            this.SubgridSize = subgridSize;
            this.KernelSize = kernelSize;

            this.MaxTimestepsPerSubgrid = maxTimesteps;

            this.ScaleArcSec = scale;
            this.ImageSize = scale * gridSize;

            this.WLayerCount = wLayerCount;
            this.WStep = wStep;

            this.GridSpheroidal = MathFunctions.CalcIdentitySpheroidal(gridSize, gridSize);
            this.SubgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridSize, subgridSize);
        }
    }
}
