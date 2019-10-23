using System;

namespace DistributedReconstruction
{
    class Program
    {
        static void Main(string[] args)
        {
            /*var total = new float[32, 32];
            for(int i = 0; i < total.GetLength(0); i++)
                for(int j = 0; j < total.GetLength(1);j++)
                    total[i, j] = i * 100 + j;

            var sections = new float[6][,];
            for(int i = 0; i < 6; i++)
            {
                var rec = SimpleDistributedReconstruction.CalculateLocalImageSection(i, 6, 32, 32);
                var x = rec.X;
                var section = SimpleDistributedReconstruction.GetImgSection(total, rec);
                section[0, 0] += 100000 * (i+1);
                sections[i] = section;
            }
            var stitched = new float[32, 32];
            SimpleDistributedReconstruction.StitchImage(sections, stitched, 6);*/
                
            //RunningMethods.RunSimulated(args);
            //RunningMethods.RunTinyMeerKAT(args);
            //RunningMethods.RunTest(args);
            RunningMethods.RunTinyMeerKAT(args);
        }
    }
}
