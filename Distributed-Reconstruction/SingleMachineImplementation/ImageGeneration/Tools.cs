using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using static Single_Reference.Common;

namespace SingleMachineRuns.ImageGeneration
{
    class Tools
    {
        public static class LMC
        {
            public static float[,] CutN132Remnant(float[,] image)
            {
                var rectangle = new Rectangle(1768, 1183, 1768 + 90, 1183 + 90);
                var output = new float[rectangle.YExtent(), rectangle.XExtent()];
                for (int y = rectangle.Y; y < rectangle.YEnd; y++)
                    for (int x = rectangle.X; x < rectangle.XEnd; x++)
                        output[y - rectangle.Y, x - rectangle.X] = image[y, x];

                return output;
            }

            public static float[,] CutCalibration(float[,] image)
            {
                var rectangle = new Rectangle(1635, 2330, 1635 + 350, 2330 + 350);
                var output = new float[rectangle.YExtent(), rectangle.XExtent()];
                for (int y = rectangle.Y; y < rectangle.YEnd; y++)
                    for (int x = rectangle.X; x < rectangle.XEnd; x++)
                        output[y - rectangle.Y, x - rectangle.X] = image[y, x];

                return output;
            }
        }
        
        public static void Mask(float[,] image, int cutFactor)
        {
            var yOffset = image.GetLength(0) / 2 - (image.GetLength(0) / cutFactor) / 2;
            var xOffset = image.GetLength(1) / 2 - (image.GetLength(1) / cutFactor) / 2;

            for (int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    if (!(y >= yOffset & y < (yOffset + image.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + image.GetLength(1) / cutFactor)))
                        image[y, x] = 0.0f;
        }

        public static void ReverseMask(float[,] image, int cutFactor)
        {
            var yOffset = image.GetLength(0) / 2 - (image.GetLength(0) / cutFactor) / 2;
            var xOffset = image.GetLength(1) / 2 - (image.GetLength(1) / cutFactor) / 2;

            for (int y = 0; y < image.GetLength(0); y++)
                for (int x = 0; x < image.GetLength(1); x++)
                    if ((y >= yOffset & y < (yOffset + image.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + image.GetLength(1) / cutFactor)))
                        image[y, x] = 0.0f;
        }

        public static void WriteToCSV(float[,] image, string file)
        {
            using(var writer = new StreamWriter(file))
            {
                var line = new float[image.GetLength(1)];
                for(int i = 0; i < image.GetLength(0);i++)
                {
                    for (int j = 0; j < image.GetLength(1); j++)
                        line[j] = image[i, j];
                    writer.WriteLine(string.Join(";", line));
                }
            }
        }

    }
}
