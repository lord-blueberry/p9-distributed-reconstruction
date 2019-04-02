using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;
using System.Numerics;

namespace Single_Machine.IDG
{
    /// <summary>
    /// hack is here to have the first version resemble the IDG reference implementation more
    /// </summary>
    public class SubgridHack
    {
        public int baselineIdx;
        public int timeSampleStart;
        public int timeSampleCount;

        public int UPixel;
        public int VPixel;
        public int WLambda;
    }

    class SubgridData
    {
        public int TimeStepCount { get; }

        public int Station0 { get; }
        public int Station1 { get; }

        public int UPixel { get; }
        public int VPixel { get; }
        public int WLambda { get; }

        public int ATermIndex { get; }

        public float UOffset { get; }
        public float VOffset { get; }
        public float WOffset { get; }

        public IList<UVWTuple> UVW { get; }
        public IList<float> Wavenumbers { get; }
        public Complex[,] Visibilities { get; }

        public SubgridData(GriddingConstants p)
        {
            

            this.UOffset = (float)((this.UPixel + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / p.ImageSize));
            this.VOffset = (float)((this.VPixel + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / p.ImageSize));

            double tmpW = p.WStepLambda * (this.WLambda + 0.5); //????
            this.WOffset = (float)(2 * PI * tmpW);

        }


        public class UVWTuple
        {
            public float U { get; }
            public float V { get; }
            public float W { get; }

            public UVWTuple(float u, float v, float w)
            {
                this.U = u;
                this.V = v;
                this.W = w;
            }
        }
    }
}
