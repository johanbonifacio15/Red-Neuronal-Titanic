namespace TitanicNeuralNetwork.Network.Training
{
    public static class PerformanceAnalyzer
    {
        public static void CompareRuns(TrainingMetrics parallelMetrics, TrainingMetrics sequentialMetrics)
        {
            Console.WriteLine("\n=== COMPARACIÓN DETALLADA ===");
            Console.WriteLine($"| Métrica          | Paralelo  | Secuencial | Mejora    |");
            Console.WriteLine($"|------------------|-----------|------------|-----------|");
            Console.WriteLine($"| Tiempo (s)       | {parallelMetrics.TrainingTime.TotalSeconds,7:F2} | {sequentialMetrics.TrainingTime.TotalSeconds,8:F2}  | {sequentialMetrics.TrainingTime.TotalSeconds / parallelMetrics.TrainingTime.TotalSeconds,6:F2}x |");
            Console.WriteLine($"| Épocas/s         | {parallelMetrics.EpochsPerSecond,7:F2} | {sequentialMetrics.EpochsPerSecond,8:F2}  | {parallelMetrics.EpochsPerSecond / sequentialMetrics.EpochsPerSecond,6:F2}x |");
            Console.WriteLine($"| Precisión final  | {parallelMetrics.BestAccuracy,7:P2} | {sequentialMetrics.BestAccuracy,8:P2}  | {parallelMetrics.BestAccuracy - sequentialMetrics.BestAccuracy,6:P2} |");

            AnalyzeEpochHistory(parallelMetrics, sequentialMetrics);
        }

        private static void AnalyzeEpochHistory(TrainingMetrics parallel, TrainingMetrics sequential)
        {
            var commonEpochs = parallel.EpochHistory.Keys
                .Intersect(sequential.EpochHistory.Keys)
                .OrderBy(e => e)
                .ToList();

            var comparisonPoints = new[] { 0, 10, 20, 30, 50, 100 }
                .Where(e => e <= commonEpochs.Max())
                .Select(e => commonEpochs.FirstOrDefault(epoch => epoch >= e))
                .Distinct()
                .ToList();

            foreach (var epoch in comparisonPoints)
            {
                if (parallel.EpochHistory.TryGetValue(epoch, out var para) &&
                    sequential.EpochHistory.TryGetValue(epoch, out var seq))
                {
                    Console.WriteLine($"\nEn época {epoch}:");
                    Console.WriteLine($"- Paralelo:  Precisión = {para.Accuracy:P2}, Pérdida = {para.Loss:F4}");
                    Console.WriteLine($"- Secuencial: Precisión = {seq.Accuracy:P2}, Pérdida = {seq.Loss:F4}");
                    Console.WriteLine($"- Diferencia: {para.Accuracy - seq.Accuracy:P2}");
                }
            }
        }
    }
}