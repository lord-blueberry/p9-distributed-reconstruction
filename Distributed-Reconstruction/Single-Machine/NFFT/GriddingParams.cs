using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Single_Machine.NFFT
{
    class GriddingParams
    {
        public float ImageSize { get; }
        public int KernelSize { get; }
        public int SubgridSize { get; }
        public int GridSize { get; }
        public float CellSize { get; }

        public int WLayerCount { get; }
        public float WStep { get; }
        public float WStepLambda { get; }

        public int MaxTimestepsPerSubgrid { get; }

        public GriddingParams(int kernelSize)
        {
            this.KernelSize = KernelSize;
        }
    }
}
