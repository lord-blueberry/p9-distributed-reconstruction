using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Numerics;

namespace Single_Machine
{
    class BinaryIO
    {
        public static void WriteImage(string file, double[,]image)
        {
            var stream = File.Create(file);
            var formatter = new BinaryFormatter();
            formatter.Serialize(stream, image);
        }

        public static void WriteUVW(string file, double[,,] uvw)
        {
            var stream = File.Create(file);
            var formatter = new BinaryFormatter();
            formatter.Serialize(stream, uvw);
        }

        public static void WriteFrequency(string file, double[] frequencies)
        {
            var stream = File.Create(file);
            var formatter = new BinaryFormatter();
            formatter.Serialize(stream, frequencies);
        }

        public static void WriteVisibilities(string file, Complex[,,] visibilities)
        {
            var stream = File.Create(file);
            var formatter = new BinaryFormatter();
            formatter.Serialize(stream, visibilities);
        }

        public static double[,,] ReadUVW(string file)
        {
            var stream = File.OpenRead(file);
            var formatter = new BinaryFormatter();
            return (double[,,])formatter.Deserialize(stream);
        }

        public static double[] ReadFrequencies(string file)
        {
            var stream = File.OpenRead(file);
            var formatter = new BinaryFormatter();
            return (double[])formatter.Deserialize(stream);
        }

        public static Complex[,,] ReadVisibilities(string file)
        {
            var stream = File.OpenRead(file);
            var formatter = new BinaryFormatter();
            return (Complex[,,])formatter.Deserialize(stream);
        }
    }
}
