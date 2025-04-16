using System;

namespace TitanicNeuralNetwork.Network
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double[] PreviousWeightUpdates { get; set; }
        public double Bias { get; set; }
        public double Output { get; set; }
        public double Delta { get; set; }

        public Neuron(int inputSize, bool useHeInitialization = false, double weightScale = 1.0)
        {
            Weights = new double[inputSize];
            PreviousWeightUpdates = new double[inputSize];

            double factor = useHeInitialization ?
                Math.Sqrt(2.0 / inputSize) * weightScale :
                0.1 * weightScale;

            for (int i = 0; i < inputSize; i++)
            {
                Weights[i] = (new Random().NextDouble() * 2 - 1) * factor;
                PreviousWeightUpdates[i] = 0;
            }
            Bias = (new Random().NextDouble() * 2 - 1) * factor * 0.1;
        }

        public double Activate(double[] inputs)
        {
            double sum = Bias;
            for (int i = 0; i < Weights.Length; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            Output = sum;
            return Output;
        }
    }
}