using System;
using System.Diagnostics;
using TitanicNeuralNetwork.Data;

namespace TitanicNeuralNetwork.Network.Training
{
    public class TrainingEngine
    {
        private readonly NeuralNetwork _network;
        private readonly DataPoint[] _trainingSet;
        private readonly DataPoint[] _testSet;
        private readonly bool _useParallel;
        private readonly int _patience;
        private const double SignificantImprovement = 0.005;
        private const int MaxEpochs = 1000;

        public TrainingEngine(NeuralNetwork network, DataPoint[] trainingSet, DataPoint[] testSet, bool useParallel)
        {
            _network = network;
            _trainingSet = trainingSet;
            _testSet = testSet;
            _useParallel = useParallel;
            _network.UseParallel = useParallel;
            _patience = useParallel ? 75 : 50;
        }

        public TrainingMetrics Train(int maxEpochs = 1000)
        {
            var metrics = new TrainingMetrics();
            var stopwatch = Stopwatch.StartNew();

            var fixedEvalPoints = new HashSet<int> { 0, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999 };

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                _network.TrainEpoch(_trainingSet);

                bool shouldEval = fixedEvalPoints.Contains(epoch) || (epoch > 100 && epoch % 25 == 0);

                if (shouldEval)
                {
                    var (loss, accuracy) = Evaluate();
                    metrics.EpochHistory[epoch] = (accuracy, loss);

                    UpdateBestMetrics(metrics, epoch, accuracy, loss);

                    Console.WriteLine($"Época {epoch,4}: Precisión = {accuracy:P2} | Pérdida = {loss:F4} | Mejor = {metrics.BestAccuracy:P2} @ época {metrics.BestEpoch}");

                    if (ShouldStopEarly(epoch, metrics.NoImprovementCount))
                    {
                        Console.WriteLine($"\nEarly stopping activado (sin mejora significativa en {_patience} evaluaciones)");
                        break;
                    }
                }
            }

            stopwatch.Stop();
            FinalizeMetrics(metrics, stopwatch.Elapsed);
            return metrics;
        }

        private (double loss, double accuracy) Evaluate()
        {
            double loss = CalculateLoss();
            double accuracy = _network.Test(_testSet);
            return (loss, accuracy);
        }

        private double CalculateLoss()
        {
            double loss = 0;
            foreach (var item in _trainingSet)
            {
                var output = _network.FeedForward(item.Inputs)[0];
                var target = item.Outputs[0];
                loss += Math.Pow(output - target, 2);
            }
            return loss / _trainingSet.Length;
        }

        private bool ShouldEvaluate(int epoch)
        {
            // Evaluar siempre en estas épocas para comparación consistente
            var fixedEvalPoints = new[] { 0, 10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999 };
            if (fixedEvalPoints.Contains(epoch)) return true;

            // Evaluación adicional cada 25 épocas después de la época 100
            return epoch > 100 && epoch % 25 == 0;
        }

        private void UpdateBestMetrics(TrainingMetrics metrics, int epoch, double accuracy, double loss)
        {
            if (accuracy > metrics.BestAccuracy + SignificantImprovement)
            {
                metrics.BestAccuracy = accuracy;
                metrics.BestEpoch = epoch;
                metrics.NoImprovementCount = 0;
                metrics.FinalLoss = loss;
            }
            else
            {
                metrics.NoImprovementCount++;
            }
        }

        private bool ShouldStopEarly(int epoch, int noImprovementCount)
        {
            return noImprovementCount >= _patience && epoch > 100;
        }

        private void FinalizeMetrics(TrainingMetrics metrics, TimeSpan elapsed)
        {
            metrics.TrainingTime = elapsed;
            metrics.TotalEpochs = metrics.EpochHistory.Keys.Max() + 1;
            metrics.CalculateDerivedMetrics();
        }
    }
}
