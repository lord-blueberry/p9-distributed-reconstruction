using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;
using System.Numerics;

namespace Core.ImageDomainGridder
{
    /// <summary>
    /// hack is here to have the first version resemble the IDG reference implementation more
    /// </summary>
    public class Subgrid
    {
        public int baselineIdx;
        public int timeSampleStart;
        public int timeSampleCount;

        public int UPixel;
        public int VPixel;
        public int WLambda;
    }
}
