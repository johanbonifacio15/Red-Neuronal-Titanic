using System;
using System.Linq;
using System.Threading.Tasks;
using TitanicNeuralNetwork.Data;

namespace TitanicNeuralNetwork.Network
{
    public class NeuralNetwork
    {
        private readonly Layer[] _layers;
        private readonly Random _random = new();

        public double LearningRate { get; set; } = 0.01;
        public bool UseParallel { get; set; } = true;
        public double Momentum { get; set; } = 0.9;

        public NeuralNetwork(int inputSize, int[] hiddenLayers, int outputSize)
        {
            ValidateNetworkArchitecture(hiddenLayers);
            _layers = InitializeLayers(inputSize, hiddenLayers, outputSize);
        }

        private static void ValidateNetworkArchitecture(int[] hiddenLayers)
        {
            if (hiddenLayers == null || hiddenLayers.Length == 0)
                throw new ArgumentException("Debe haber al menos una capa oculta");
        }

        private static Layer[] InitializeLayers(int inputSize, int[] hiddenLayers, int outputSize)
        {
            var layers = new Layer[hiddenLayers.Length + 1];

            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                int prevSize = i == 0 ? inputSize : hiddenLayers[i - 1];
                layers[i] = new Layer(prevSize, hiddenLayers[i], useHeInitialization: true);
            }

            layers[^1] = new Layer(hiddenLayers.Last(), outputSize, isOutputLayer: true, weightScale: 0.5);
            return layers;
        }

        public double[] FeedForward(double[] inputs)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs), "Input data cannot be null");

            return _layers.Aggregate(inputs, (current, layer) => layer.Activate(current));
        }

        public void TrainEpoch(DataPoint[] trainingData)
        {
            if (UseParallel)
            {
                Parallel.For(0, trainingData.Length, i => ProcessTrainingItem(trainingData[i]));
            }
            else
            {
                foreach (var data in trainingData)
                {
                    ProcessTrainingItem(data);
                }
            }
        }

        private void ProcessTrainingItem(DataPoint data)
        {
            var outputs = FeedForward(data.Inputs);
            Backpropagate(outputs, data.Outputs, data.Inputs);
        }

        public void Backpropagate(double[] outputs, double[] targets, double[] inputs)
        {
            CalculateOutputLayerDeltas(outputs, targets);
            CalculateHiddenLayersDeltas();
            UpdateWeights(inputs);
        }

        private void CalculateOutputLayerDeltas(double[] outputs, double[] targets)
        {
            for (int i = 0; i < _layers[^1].Neurons.Length; i++)
            {
                var neuron = _layers[^1].Neurons[i];
                neuron.Delta = (outputs[i] - targets[i]) * Activation.SigmoidDerivative(neuron.Output);
            }
        }

        private void CalculateHiddenLayersDeltas()
        {
            for (int l = _layers.Length - 2; l >= 0; l--)
            {
                for (int i = 0; i < _layers[l].Neurons.Length; i++)
                {
                    var neuron = _layers[l].Neurons[i];
                    neuron.Delta = CalculateNeuronDelta(l, i, neuron);
                }
            }
        }

        private double CalculateNeuronDelta(int layerIndex, int neuronIndex, Neuron neuron)
        {
            double error = 0;
            var nextLayer = _layers[layerIndex + 1];

            for (int j = 0; j < nextLayer.Neurons.Length; j++)
            {
                error += nextLayer.Neurons[j].Delta * nextLayer.Neurons[j].Weights[neuronIndex];
            }

            return error * Activation.LeakyReLUDerivative(neuron.Output);
        }

        private void UpdateWeights(double[] inputs)
        {
            for (int l = 0; l < _layers.Length; l++)
            {
                var prevOutputs = l == 0 ? inputs : _layers[l - 1].Outputs;

                for (int i = 0; i < _layers[l].Neurons.Length; i++)
                {
                    UpdateNeuronWeights(_layers[l].Neurons[i], prevOutputs);
                }
            }
        }

        private void UpdateNeuronWeights(Neuron neuron, double[] prevOutputs)
        {
            for (int j = 0; j < neuron.Weights.Length; j++)
            {
                double weightUpdate = -LearningRate * neuron.Delta * prevOutputs[j];
                neuron.Weights[j] += weightUpdate + (Momentum * neuron.PreviousWeightUpdates[j]);
                neuron.PreviousWeightUpdates[j] = weightUpdate;
            }

            neuron.Bias -= LearningRate * neuron.Delta;
        }

        public double Test(DataPoint[] testData)
        {
            int correct = testData.Count(data => IsPredictionCorrect(data));
            return (double)correct / testData.Length;
        }

        private bool IsPredictionCorrect(DataPoint data)
        {
            var output = FeedForward(data.Inputs)[0];
            var predicted = output > 0.5 ? 1 : 0;
            var actual = data.Outputs[0] > 0.5 ? 1 : 0;
            return predicted == actual;
        }
    }
}