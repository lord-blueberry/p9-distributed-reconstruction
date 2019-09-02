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

            public int YEnd { get; private set; }
            public int XEnd { get; private set; }

            public Rectangle(int y, int x, int yEnd, int xEnd)
            {
                Y = y;
                X = x;
                YEnd = yEnd;
                XEnd = xEnd;
            }
        }



    }
}
