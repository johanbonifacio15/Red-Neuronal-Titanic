using System;
using System.Collections.Generic;

namespace TitanicNeuralNetwork.Network.Training
{
    public class TrainingMetrics
    {
        public TimeSpan TrainingTime { get; set; }
        public int TotalEpochs { get; set; }
        public double BestAccuracy { get; set; }
        public int BestEpoch { get; set; }
        public double FinalLoss { get; set; }
        public double SpeedupFactor { get; set; }
        public double EpochsPerSecond { get; set; }
        public int NoImprovementCount { get; set; }
        public Dictionary<int, (double Accuracy, double Loss)> EpochHistory { get; } = new();

        public void CalculateDerivedMetrics(TimeSpan sequentialTime = default)
        {
            EpochsPerSecond = TotalEpochs / TrainingTime.TotalSeconds;

            if (sequentialTime != default)
            {
                SpeedupFactor = sequentialTime.TotalSeconds / TrainingTime.TotalSeconds;
            }
        }

        public void PrintSummary()
        {
            Console.WriteLine("\n=== RESUMEN DE ENTRENAMIENTO ===");
            Console.WriteLine($"- Tiempo total: {TrainingTime.TotalSeconds:F2}s");
            Console.WriteLine($"- Épocas ejecutadas: {TotalEpochs}");
            Console.WriteLine($"- Épocas/segundo: {EpochsPerSecond:F2}");
            Console.WriteLine($"- Mejor precisión: {BestAccuracy:P2} (época {BestEpoch})");
            Console.WriteLine($"- Pérdida final: {FinalLoss:F4}");

            if (SpeedupFactor > 0)
            {
                Console.WriteLine($"- Factor de aceleración: {SpeedupFactor:F2}x");
            }
        }
    }
}
