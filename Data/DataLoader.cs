using System;
using System.IO;
using System.Linq;

namespace TitanicNeuralNetwork.Data
{
    public static class DataLoader
    {
        private const int ExpectedColumns = 12;
        private const int ExpectedFeatures = 15;
        private const double DefaultTrainRatio = 0.8;
        private const double MaxAge = 80;
        private const double MaxFare = 512;

        public static (DataPoint[] trainingSet, DataPoint[] testSet) LoadAndSplitData(string path, double trainRatio = DefaultTrainRatio)
        {
            var dataPath = GetDataPath(path);
            var lines = ReadAndValidateFile(dataPath);
            var stats = CalculateStatistics(lines);

            return ProcessAndSplitData(lines, stats, trainRatio);
        }

        private static string GetDataPath(string path)
        {
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "titanic.csv");
            if (!File.Exists(dataPath))
            {
                throw new FileNotFoundException($"No se encontró el archivo CSV en: {dataPath}");
            }
            return dataPath;
        }

        private static IEnumerable<string> ReadAndValidateFile(string dataPath)
        {
            var lines = File.ReadAllLines(dataPath).Skip(1); 
            if (!lines.Any())
            {
                throw new InvalidDataException("El archivo CSV no contiene datos");
            }
            return lines;
        }

        private static (DataPoint[] trainingSet, DataPoint[] testSet) ProcessAndSplitData(
            IEnumerable<string> lines, (double AvgAge, double AvgFare) stats, double trainRatio)
        {
            var random = new Random();
            var allData = lines
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line => CreateDataPoint(line, stats))
                .Where(x => x != null)
                .OrderBy(_ => random.Next())
                .ToArray();

            ValidateFeatureCount(allData);
            return SplitData(allData, trainRatio);
        }

        private static DataPoint CreateDataPoint(string line, (double AvgAge, double AvgFare) stats)
        {
            var values = line.Split(',');
            if (values.Length < ExpectedColumns)
            {
                Console.WriteLine($"Línea ignorada (columnas insuficientes): {line}");
                return null;
            }

            try
            {
                var inputs = ProcessFeatures(values, stats);
                var survived = values[1].Trim() == "1" ? 1.0 : 0.0;
                return new DataPoint(inputs, new[] { survived });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error procesando línea: {line}\n{ex.Message}");
                return null;
            }
        }

        private static double[] ProcessFeatures(string[] values, (double AvgAge, double AvgFare) stats)
        {
            // Preprocesamiento de datos
            double pclass = (4 - ParseDouble(values[2])) / 3.0;
            double sex = values[4].Trim().ToLower() == "male" ? 1.0 : 0.0;

            double age = string.IsNullOrWhiteSpace(values[5]) ? stats.AvgAge : ParseDouble(values[5]);
            age = Normalize(age, 0, MaxAge);

            double sibsp = Math.Log(1 + ParseDouble(values[6]));
            double parch = Math.Log(1 + ParseDouble(values[7]));

            double fare = string.IsNullOrWhiteSpace(values[9]) ? stats.AvgFare : ParseDouble(values[9]);
            fare = Normalize(Math.Log(1 + fare), 0, Math.Log(1 + MaxFare));

            string title = ExtractTitle(values[3]);
            double isAlone = (ParseDouble(values[6]) + ParseDouble(values[7])) == 0 ? 1.0 : 0.0;
            double ageClass = pclass * (1 - age);
            double familySize = ParseDouble(values[6]) + ParseDouble(values[7]) + 1;
            double farePerPerson = fare / familySize;

            return new[] {
            pclass, sex, age, sibsp, parch, fare,
            isAlone, ageClass, farePerPerson, string.IsNullOrWhiteSpace(values[10]) ? 0.0 : 1.0,
            title == "Mr" ? 1.0 : 0.0, title == "Mrs" ? 1.0 : 0.0,
            title == "Master" ? 1.0 : 0.0, title == "Miss" ? 1.0 : 0.0,
            (values[11].Trim() != "C" && values[11].Trim() != "Q" && values[11].Trim() != "S") ? 1.0 : 0.0
        };
        }

        private static void ValidateFeatureCount(DataPoint[] data)
        {
            if (data.Any(d => d.Inputs.Length != ExpectedFeatures))
            {
                var badData = data.Where(d => d.Inputs.Length != ExpectedFeatures).ToList();
                Console.WriteLine($"\nERROR: Se encontraron {badData.Count} datos con tamaño incorrecto");
                throw new InvalidDataException($"Todos los datos deben tener exactamente {ExpectedFeatures} características");
            }
        }

        private static (DataPoint[] trainingSet, DataPoint[] testSet) SplitData(DataPoint[] allData, double trainRatio)
        {
            int trainSize = (int)(allData.Length * trainRatio);
            return (allData.Take(trainSize).ToArray(), allData.Skip(trainSize).ToArray());
        }

        private static (double AvgAge, double AvgFare) CalculateStatistics(IEnumerable<string> lines)
        {
            double totalAge = 0, totalFare = 0;
            int ageCount = 0, fareCount = 0;

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                var values = line.Split(',');
                if (values.Length < 12) continue;

                // Procesar edad
                if (!string.IsNullOrWhiteSpace(values[5]) && double.TryParse(values[5], out double age))
                {
                    totalAge += age;
                    ageCount++;
                }

                // Procesar fare
                if (!string.IsNullOrWhiteSpace(values[9]) && double.TryParse(values[9], out double fare))
                {
                    totalFare += fare;
                    fareCount++;
                }
            }

            return (totalAge / Math.Max(1, ageCount), totalFare / Math.Max(1, fareCount));
        }

        private static string ExtractTitle(string name)
        {
            var match = System.Text.RegularExpressions.Regex.Match(name, @"\b([A-Za-z]+)\.");
            if (match.Success)
            {
                string title = match.Groups[1].Value;
                // Consolidar títulos raros
                if (title == "Mlle" || title == "Ms") return "Miss";
                if (title == "Mme") return "Mrs";
                if (title == "Don" || title == "Rev" || title == "Dr" ||
                    title == "Major" || title == "Sir" || title == "Col" ||
                    title == "Capt" || title == "Jonkheer") return "Rare";
                return title;
            }
            return "";
        }

        private static double ParseDouble(string value)
        {
            if (double.TryParse(value, out double result))
            {
                return result;
            }
            return 0.0;
        }

        private static double Normalize(double value, double min, double max)
        {
            // Asegurar que el valor esté dentro del rango
            value = Math.Max(min, Math.Min(max, value));
            return (value - min) / (max - min);
        }
    }
}