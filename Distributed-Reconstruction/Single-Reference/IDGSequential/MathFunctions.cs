using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Single_Reference.IDGSequential
{
    class MathFunctions
    {
        public const double SPEED_OF_LIGHT = 299792458.0; //todo: get more accurate speed of light constant

        #region Spheriodal
        public static float EvaluateSpheroidal(float nu)
        {
            float[,] P = {
                { 8.203343e-2f, -3.644705e-1f, 6.278660e-1f, -5.335581e-1f, 2.312756e-1f },
                {4.028559e-3f, -3.697768e-2f, 1.021332e-1f, -1.201436e-1f, 6.412774e-2f}};
            float[,] Q ={
                {1.0000000e0f, 8.212018e-1f, 2.078043e-1f},
                {1.0000000e0f, 9.599102e-1f, 2.918724e-1f}};

            int part;
            float end;
            if (nu >= 0.0 && nu < 0.75)
            {
                part = 0;
                end = 0.75f;
            }
            else if (nu >= 0.75 && nu <= 1.00)
            {
                part = 1;
                end = 1.0f;
            }
            else
            {
                return 0.0f;
            }

            float nusq = nu * nu;
            float delnusq = nusq - end * end;
            float delnusqPow = delnusq;
            float top = P[part, 0];
            for (var k = 1; k < 5; k++)
            {
                top += P[part, k] * delnusqPow;
                delnusqPow *= delnusq;
            }

            float bot = Q[part, 0];
            delnusqPow = delnusq;
            for (var k = 1; k < 3; k++)
            {
                bot += Q[part, k] * delnusqPow;
                delnusqPow *= delnusq;
            }

            if (bot == 0.0f)
            {
                return 0.0f;
            }
            else
            {
                return (1.0f - nusq) * (top / bot);
            }

        }

        public static float[,] CalcIdentitySpheroidal(int height, int width)
        {
            float[,] output = new float[height, width];
            float val = 1.0f;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    output[i, j] = val;

            return output;
        }

        public static float[,] CalcExampleSpheroidal(int height, int width)
        {
            float[,] output = new float[height, width];

            //evaluate rows
            float[] y =  new float[height];
            for (int i = 0; i < height; i++)
            {
                float tmp = System.Math.Abs(-1 + i * 2.0f / height);
                y[i] = EvaluateSpheroidal(tmp);
            }

            // Evaluate columns
            float[] x = new float[width];
            for (int i = 0; i < width; i++)
            {
                float tmp = System.Math.Abs(-1 + i * 2.0f / width);
                x[i] = EvaluateSpheroidal(tmp);
            }

            for(int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    output[i, j] = y[i] * x[j];
                
            return output;
        }

        public static double[] FrequencyToWavenumber(double[] frequencies)
        {
            var output = new double[frequencies.Length];
            for (int i = 0; i < output.Length; i++)
                output[i] = (2.0f * System.Math.PI * frequencies[i] / SPEED_OF_LIGHT);
            return output;
        }
        #endregion
    }
}
