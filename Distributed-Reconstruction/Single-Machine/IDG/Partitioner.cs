using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Single_Machine.IDG
{
    public class Partitioner
    {
        /// <summary>
        /// Partitions the Visibility data into subgrids. 
        /// 
        /// In the original implementation, this is called "Plan"
        /// </summary>
        /// <param name="c"></param>
        /// <param name="uvw"></param>
        /// <param name="frequencies"></param>
        /// <returns></returns>
        public static List<List<SubgridHack>> CreatePartition(GriddingConstants c, double[,,] uvw, double[] frequencies)
        {
            var imagesize = c.CellSize * c.GridSize;
            List<List<SubgridHack>> outputGrids = new List<List<SubgridHack>>(uvw.GetLength(0));
            for(int baseline = 0; baseline < uvw.GetLength(0); baseline++)
            {
                var baselineOutput = new List<SubgridHack>(uvw.GetLength(1));
                outputGrids.Add(baselineOutput);
                //idg checks a-terms bins we ignore the a-term correction here, so we can simplify and only iterate over 

                Datapoint[,] datapoints = new Datapoint[uvw.GetLength(1), frequencies.Length]; //timeSamplesCount  channelsCount 
                int time = 0;
                for (time = 0; time < uvw.GetLength(1); time++)
                {
                    //convert visibilities
                    for (int channel = 0; channel < frequencies.Length; channel++)
                    {
                        var dp = new Datapoint
                        {
                            timestep = time,
                            channel = channel,
                            uPixel = MetersToPixels(uvw[baseline, time, 0], imagesize, frequencies[channel]),
                            vPixel = MetersToPixels(uvw[baseline, time, 1], imagesize, frequencies[channel]),
                            wLambda = MetersToLambda(uvw[baseline, time, 2], frequencies[channel])
                        };
                        datapoints[time, channel] = dp;
                    }
                }

                time = 0;
                Subgrid subgrid = new Subgrid(c);
                while (time < uvw.GetLength(1))
                {
                    subgrid.Reset();
                    int timeSamplePerSubgrid = 0;

                    //this is taken from the original IDG implementation. Here may be room for simplification
                    for (; time < uvw.GetLength(1); time++)
                    {
                        var dpChannel0 = datapoints[time, 0];
                        var dpChannelMax = datapoints[time, frequencies.Length - 1];
                        var hack = dpChannelMax.Copy();
                        hack.wLambda = dpChannel0.wLambda;  // hack taken over from IDG reference implementation

                        if (subgrid.AddVisibility(dpChannel0) && subgrid.AddVisibility(hack))
                        {
                            timeSamplePerSubgrid++;
                            if (timeSamplePerSubgrid == c.MaxTimestepsPerSubgrid)
                                break;
                        }
                        else
                        {
                            break;
                        }
                    }

                    //Handle empty subgrids
                    if (timeSamplePerSubgrid == 0)
                    {
                        var dp = datapoints[time, 0];

                        if (Double.IsInfinity(dp.uPixel) && Double.IsInfinity(dp.vPixel))
                        {
                            throw new Exception("could not place (all) visibilities on subgrid (too many channnels, kernel size too large)");
                        }
                        else if (Double.IsInfinity(dp.uPixel) || Double.IsInfinity(dp.vPixel))
                        {
                            //added by me
                            throw new Exception("could not place (all) visibilities on subgrid (too many channnels, kernel size too large)");
                        }
                        else
                        {
                            // Advance to next timeslot when visibilities for current timeslot had infinite coordinates
                            time++;
                            continue;
                        }
                    }

                    subgrid.Finish();
                    if (subgrid.InRange())
                    {
                        //TODO: Fix hack and hand over data properly
                        var data = new SubgridHack();
                        data.timeSampleCount = timeSamplePerSubgrid;
                        data.timeSampleStart = time - timeSamplePerSubgrid;
                        data.baselineIdx = baseline;
                        data.UPixel = subgrid.UPixel;
                        data.VPixel = subgrid.VPixel;
                        data.WLambda = subgrid.WIndex;

                        baselineOutput.Add(data);
                    }
                    else
                    {
                        subgrid.InRange();
                        throw new Exception("subgrid falls not within grid");
                    }

                }
            }//baseline

            return outputGrids;
        }

        private static double MetersToPixels(double meters, double imageSize, double frequency) => meters * imageSize * (frequency / Math.SPEED_OF_LIGHT);

        private static double MetersToLambda(double meters, double frequency) => meters * (frequency / Math.SPEED_OF_LIGHT);
        
        
        private class Datapoint
        {
            public int timestep;
            public int channel;
            public double uPixel;
            public double vPixel;
            public double wLambda;

            public Datapoint Copy()
            {
                var copy = new Datapoint();
                copy.timestep = this.timestep;
                copy.channel = this.channel;
                copy.uPixel = this.uPixel;
                copy.vPixel = this.vPixel;
                copy.wLambda = this.wLambda;
                return copy;
            }
        }
        /// <summary>
        /// implementation taken from original IDG. This class can potentially be made obsolete by refactoring CreatePlan()
        /// </summary>
        private class Subgrid
        {
            private GriddingConstants param;
            private double uMin;
            private double uMax;
            private double vMin;
            private double vMax;
            private double uvWith;
            
            private bool finished;

            public int UPixel { get; set; }
            public int VPixel { get; set; }
            public int WIndex { get; set; }

            public Subgrid(GriddingConstants p)
            {
                this.param = p;
                this.Reset();
            }


            public void Reset()
            {
                uMin = Double.PositiveInfinity;
                uMax = Double.NegativeInfinity;
                vMin = Double.PositiveInfinity;
                vMax = Double.NegativeInfinity;
                uvWith = 0;
                WIndex = 0;
                finished = false;

                UPixel = 0;
                VPixel = 0;
            }

            public bool AddVisibility(Datapoint vis)
            {
                if (Double.IsInfinity(vis.uPixel) || Double.IsInfinity(vis.vPixel))
                    return false;

                int wIndex_ = 0;
                if (param.WStep != 0.0)
                    wIndex_ = (int)System.Math.Floor(vis.wLambda / param.WStep);

                // if this is not the first sample, it should map to the
                // same w_index as the others, if not, return false
                if (Double.IsInfinity(this.uMin) && this.WIndex != wIndex_)
                    return false;

                //compute candidate values
                double uMin_ = System.Math.Min(uMin, vis.uPixel);
                double uMax_ = System.Math.Max(uMax, vis.uPixel);
                double vMin_ = System.Math.Min(vMin, vis.vPixel);
                double vMax_ = System.Math.Max(vMax, vis.vPixel);
                double uvWidth_ = System.Math.Max(uMax_ - uMin_, vMax_ - vMin_);

                //check if visibility fits 
                if((uvWidth_ + param.KernelSize) >= param.SubgridSize) {
                    return false;
                }
                else
                {
                    uMin = uMin_;
                    uMax = uMax_;
                    vMin = vMin_;
                    vMax = vMax_;
                    uvWith = uvWidth_;
                    WIndex = wIndex_;
                    return true;
                }
            }

            public bool InRange()
            {
                int uvMaxPixels = System.Math.Max(UPixel, VPixel);
                int uvMinPixels = System.Math.Min(UPixel, VPixel);

                // Return whether the subgrid fits in grid and w-stack
                //TODO: HACKED value here, uvMinPixels >= 0
                return uvMinPixels >= 1 &&
                    uvMaxPixels <= (param.GridSize - param.SubgridSize) &&
                    WIndex >= -param.WLayerCount &&
                    WIndex < param.WLayerCount;
            }

            private void ComputeCoordinates()
            {
                UPixel = (int)System.Math.Round((uMax + uMin) / 2);
                VPixel = (int)System.Math.Round((vMax + vMin) / 2);

                // Shift center from middle of grid to top left
                UPixel += (param.GridSize / 2);
                VPixel += (param.GridSize / 2);

                // Shift from middle of subgrid to top left
                UPixel -= (param.SubgridSize) / 2;
                VPixel -= (param.SubgridSize) / 2;
            }

            public void Finish()
            {
                finished = true;
                this.ComputeCoordinates();
            }
        }
    }
}
