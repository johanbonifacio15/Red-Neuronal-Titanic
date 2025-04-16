using System.Linq;

namespace TitanicNeuralNetwork.Network
{
    public class Layer
    {
        public Neuron[] Neurons { get; }
        public double[] Outputs { get; private set; }
        public bool IsOutputLayer { get; }

        public Layer(int inputSize, int neuronCount, bool useHeInitialization = false,
                    bool isOutputLayer = false, double weightScale = 1.0)
        {
            Neurons = new Neuron[neuronCount];
            IsOutputLayer = isOutputLayer;

            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(inputSize, useHeInitialization, weightScale);
            }
        }

        public double[] Activate(double[] inputs)
        {
            Outputs = Neurons.Select(neuron =>
            {
                double activation = neuron.Activate(inputs);
                return IsOutputLayer ? Activation.Sigmoid(activation) : Activation.LeakyReLU(activation);
            }).ToArray();

            return Outputs;
        }
    }
}