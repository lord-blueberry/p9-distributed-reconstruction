using System;
using System.Collections.Generic;
using System.Text;

namespace Single_Reference.Deconvolution
{
    public class CommonMethods
    {
        public static double ShrinkElasticNet(double value, double lambda, double alpha)
        {
            //ShrinkPositive(value, lambda * alpha) / (1 + lambda * (1 - alpha));
            return 0.0;
        }
    }
}
