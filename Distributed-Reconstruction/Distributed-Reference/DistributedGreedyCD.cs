using System;
using System.Collections.Generic;
using System.Text;

namespace Distributed_Reference
{
    class DistributedGreedyCD
    {
        public class Rectangle
        {
            public int X { get; private set; }
            public int Y { get; private set; }
            public int XLength { get; private set; }
            public int YLength { get; private set; }

            public Rectangle(int x, int y, int xLen, int yLen)
            {
                X = x;
                Y = y;
                XLength = xLen;
                YLength = yLen;
            }
        }

        public static bool Deconvolve(double[,] xImage, double[,] res, double[,] psf, double lambda, double alpha, Rectangle rec, int maxIteration = 100, double[,] dirtyCopy = null)
        {

            return true;
        }
    }
}
