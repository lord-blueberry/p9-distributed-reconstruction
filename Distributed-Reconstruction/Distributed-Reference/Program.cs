﻿using System;

namespace Distributed_Reference
{
    class Program
    {
        static void Main(string[] args)
        {
            var rank = 2;
            var rankCount = 10;
            //var data = DistributedData.LoadTinyMeerKAT2(rank, rankCount, @"C:\dev\GitHub\p9-data\large\fits\meerkat_tiny\");

            //RunningMethods.RunSimulated(args);
            //RunningMethods.RunTinyMeerKAT(args);
            //RunningMethods.RunTest(args);
            RunningMethods.RunTinyMeerKAT(args);
        }
    }
}
