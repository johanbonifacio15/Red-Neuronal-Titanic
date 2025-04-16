using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TitanicNeuralNetwork.Data;
using TitanicNeuralNetwork.Network;
using TitanicNeuralNetwork.Network.Training;

namespace TitanicNeuralNetwork
{
    class Program
    {
        private const int InputSize = 15;
        private static readonly int[] HiddenLayers = new[] { 10, 5 };
        private const int OutputSize = 1;
        private const double LearningRate = 0.007;
        private const double Momentum = 0.9;

        static void Main(string[] args)
        {
            Console.WriteLine("Titanic Neural Network - Predicción de Supervivencia");
            Console.WriteLine("===================================================");

            try
            {
                var (trainingSet, testSet) = LoadData();
                ShowMenu(trainingSet, testSet);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError durante la ejecución: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        private static (DataPoint[] trainingSet, DataPoint[] testSet) LoadData()
        {
            var dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "titanic.csv");
            Console.WriteLine($"Cargando datos desde: {dataPath}");

            var (trainingSet, testSet) = DataLoader.LoadAndSplitData(dataPath);

            Console.WriteLine($"\nDatos cargados correctamente:");
            Console.WriteLine($"- Ejemplos de entrenamiento: {trainingSet.Length}");
            Console.WriteLine($"- Ejemplos de prueba: {testSet.Length}");
            Console.WriteLine($"- Características por dato: {trainingSet[0].Inputs.Length}");

            return (trainingSet, testSet);
        }

        private static void ShowMenu(DataPoint[] trainingSet, DataPoint[] testSet)
        {
            Console.WriteLine("\nSeleccione el modo de entrenamiento:");
            Console.WriteLine("1. Entrenamiento Paralelo");
            Console.WriteLine("2. Entrenamiento Secuencial");
            Console.WriteLine("3. Comparar ambos modos");
            Console.Write("Ingrese su elección (1-3): ");

            var choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    TrainAndEvaluate(trainingSet, testSet, true);
                    break;
                case "2":
                    TrainAndEvaluate(trainingSet, testSet, false);
                    break;
                case "3":
                    CompareTrainingModes(trainingSet, testSet);
                    break;
                default:
                    Console.WriteLine("Opción no válida. Usando modo paralelo por defecto.");
                    TrainAndEvaluate(trainingSet, testSet, true);
                    break;
            }
        }

        static void TrainAndEvaluate(DataPoint[] trainingSet, DataPoint[] testSet, bool useParallel)
        {
            Console.WriteLine("\nConfigurando red neuronal...");
            var network = CreateNetwork();

            Console.WriteLine($"\nIniciando entrenamiento ({(useParallel ? "PARALELO" : "SECUENCIAL")})...");
            var trainer = new TrainingEngine(network, trainingSet, testSet, useParallel);
            var metrics = trainer.Train();

            metrics.PrintSummary();
            PrintExamplePredictions(network, testSet);
        }

        static void CompareTrainingModes(DataPoint[] trainingSet, DataPoint[] testSet)
        {
            Console.WriteLine("\n=== COMPARACIÓN COMPLETA ===");

            // Entrenamiento paralelo
            Console.WriteLine("\n[1/2] ENTRENAMIENTO PARALELO");
            var parallelNetwork = CreateNetwork();
            var parallelTrainer = new TrainingEngine(parallelNetwork, trainingSet, testSet, true);
            var parallelMetrics = parallelTrainer.Train(); 

            // Entrenamiento secuencial
            Console.WriteLine("\n[2/2] ENTRENAMIENTO SECUENCIAL");
            var sequentialNetwork = CreateNetwork();
            var sequentialTrainer = new TrainingEngine(sequentialNetwork, trainingSet, testSet, false);
            var sequentialMetrics = sequentialTrainer.Train(); 

            // Comparación
            PerformanceAnalyzer.CompareRuns(parallelMetrics, sequentialMetrics);
        }

        private static NeuralNetwork CreateNetwork()
        {
            var network = new NeuralNetwork(InputSize, HiddenLayers, OutputSize)
            {
                LearningRate = LearningRate,
                Momentum = Momentum
            };

            Console.WriteLine($"- Arquitectura: {InputSize}-{string.Join("-", HiddenLayers)}-{OutputSize}");
            Console.WriteLine($"- Learning rate: {network.LearningRate}");
            Console.WriteLine($"- Momentum: {network.Momentum}");

            return network;
        }

        private static void PrintExamplePredictions(NeuralNetwork network, DataPoint[] testSet, int exampleCount = 3)
        {
            Console.WriteLine("\nPredicciones de ejemplo:");
            var random = new Random();
            var examples = testSet.OrderBy(x => random.Next()).Take(exampleCount);

            foreach (var example in examples)
            {
                var output = network.FeedForward(example.Inputs)[0];
                var predicted = output > 0.5 ? "Sobrevive" : "No sobrevive";
                var actual = example.Outputs[0] > 0.5 ? "Sobrevivió" : "No sobrevivió";

                Console.WriteLine($"\n- Real: {actual}, Predicción: {predicted} ({output:F4})");
                Console.WriteLine($"  Correcto: {(predicted.StartsWith(actual.Substring(0, 4)) ? "✓" : "✗")}");
            }
        }
    }
}