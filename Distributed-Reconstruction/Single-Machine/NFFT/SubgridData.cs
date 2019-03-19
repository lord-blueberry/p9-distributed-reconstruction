using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Single_Machine.NFFT
{
    class SubgridData
    {
        public int TimeStepCount { get; }

        public int Station0 { get; }
        public int Station1 { get; }

        public int XCoordinate { get; }
        public int YCoordinate { get; }
        public int ZCoordinate { get; }


        public int ATermIndex { get; }

        public float UOffset { get; }
        public float VOffset { get; }
        public float WOffset { get; }


        public IList<UVWTuple> UVW { get; }
        public IList<float> Wavenumbers { get; }
        public Visibility[,] Visibilities { get; }

        public SubgridData(GriddingParams p)
        {
            

            this.UOffset = (float)((this.XCoordinate + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / p.ImageSize));
            this.VOffset = (float)((this.YCoordinate + p.SubgridSize / 2 - p.GridSize / 2) * (2 * PI / p.ImageSize));

            double tmpW = p.WStepLambda * (this.ZCoordinate + 0.5); //????
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

        public class Visibility
        {
            public float Real { get; }
            public float Imag { get; }

            public Visibility(float real, float imag)
            {
                this.Real = real;
                this.Imag = imag;
            }
        }
    }
}
