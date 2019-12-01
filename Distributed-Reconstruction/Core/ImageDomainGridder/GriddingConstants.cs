using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Core.ImageDomainGridder
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
        public double WStep { get; }
        public float WStepLambda { get; }

        public float[,] SubgridSpheroidal { get; }
        public float[,] GridSpheroidal { get; }

        public long VisibilitiesCount { get; }


        public GriddingConstants(long visCount, int gridSize, int subgridSize, int kernelSize, int maxTimesteps, float scaleArcSec, int wLayerCount, double wStep)
        {
            this.VisibilitiesCount = visCount;
            this.GridSize = gridSize;
            this.SubgridSize = subgridSize;
            this.KernelSize = kernelSize;

            this.MaxTimestepsPerSubgrid = maxTimesteps;

            this.ScaleArcSec = scaleArcSec;
            this.ImageSize = scaleArcSec * gridSize;

            this.WLayerCount = wLayerCount;
            this.WStep = wStep;

            this.GridSpheroidal = MathFunctions.CalcIdentitySpheroidal(gridSize, gridSize);
            this.SubgridSpheroidal = MathFunctions.CalcIdentitySpheroidal(subgridSize, subgridSize);


        }

    }
}
