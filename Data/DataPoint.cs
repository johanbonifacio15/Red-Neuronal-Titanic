namespace TitanicNeuralNetwork.Data
{
    public class DataPoint
    {
        public double[] Inputs { get; set; }
        public double[] Outputs { get; set; }

        public DataPoint(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}