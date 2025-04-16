namespace TitanicNeuralNetwork.Network
{
    public static class Activation
    {
        public static double LeakyReLU(double x, double alpha = 0.01)
        {
            return x > 0 ? x : alpha * x;
        }

        public static double LeakyReLUDerivative(double x, double alpha = 0.01)
        {
            return x > 0 ? 1.0 : alpha;
        }

        public static double Sigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            double sig = Sigmoid(x);
            return sig * (1.0 - sig);
        }
    }
}