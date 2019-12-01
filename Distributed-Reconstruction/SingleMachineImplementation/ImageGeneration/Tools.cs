using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using static Core.Common;

namespace SingleMachineRuns.ImageGeneration
{
    class Tools
    {
        public static class LMC
        {
            public static Tuple<float[,], int,int> CutN132Remnant(float[,] image)
            {
                var yOffset = 1765;
                var xOffset = 1180;
                var rectangle = new Rectangle(yOffset, xOffset, yOffset + 95, xOffset + 95);
                var output = new float[rectangle.YExtent(), rectangle.XExtent()];
                for (int y = rectangle.Y; y < rectangle.YEnd; y++)
                    for (int x = rectangle.X; x < rectangle.XEnd; x++)
                        output[y - rectangle.Y, x - rectangle.X] = image[y, x];

                return new Tuple<float[,], int, int>(output, yOffset, xOffset);
            }

            public static Tuple<float[,], int, int> CutCalibration(float[,] image)
            {
                var yOffset = 1615;
                var xOffset = 2315;
                var rectangle = new Rectangle(yOffset, xOffset, yOffset + 370, xOffset + 370);
                var output = new float[rectangle.YExtent(), rectangle.XExtent()];
                for (int y = rectangle.Y; y < rectangle.YEnd; y++)
                    for (int x = rectangle.X; x < rectangle.XEnd; x++)
                        output[y - rectangle.Y, x - rectangle.X] = image[y, x];

                return new Tuple<float[,], int, int>(output, yOffset, xOffset); 
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
                    if (!(!(y >= yOffset & y < (yOffset + image.GetLength(0) / cutFactor)) | !(x >= xOffset & x < (xOffset + image.GetLength(1) / cutFactor))))
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

        public static void WriteToMeltCSV(float[,] image, string file, int yOffset = 0, int xOffset = 0)
        {
            using (var writer = new StreamWriter(file))
            {
                writer.WriteLine("y;x;intensity");
                for (int i = 0; i < image.GetLength(0); i++)
                {
                    for (int j = 0; j < image.GetLength(1); j++)
                    {
                        var y = i + yOffset;
                        var x = j + xOffset;
                        writer.WriteLine(y + ";" + x + ";" + image[i, j]);
                    }
                        
                }
            }
        }

    }
}
