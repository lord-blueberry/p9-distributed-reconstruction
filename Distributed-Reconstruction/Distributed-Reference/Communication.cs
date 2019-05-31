using System;
using System.Collections.Generic;
using System.Text;

namespace Distributed_Reference
{
    class Communication
    {
        public class Rectangle
        {
            public int Y { get; private set; }
            public int X { get; private set; }

            public int YLength { get; private set; }
            public int XLength { get; private set; }

            public Rectangle(int y, int x, int yLen, int xLen)
            {
                Y = y;
                X = x;
                YLength = yLen;
                XLength = xLen;
            }
        }

        [Serializable]
        public class Candidate
        {
            public double XDiffMax { get; private set; }
            public double XDiff { get; private set; }

            public int YPixel { get; private set; }
            public int XPixel { get; private set; }


            public Candidate(double o, double xDiff, int y, int x)
            {
                XDiffMax = o;
                XDiff = xDiff;
                YPixel = y;
                XPixel = x;
            }
        }
    }
}
