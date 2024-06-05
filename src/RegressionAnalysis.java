import java.util.Arrays;
import java.util.stream.DoubleStream;

public class RegressionAnalysis {

    static double[] hoursStudied = {7, 4, 8, 5, 7, 3, 7, 8, 5};
    static double[] performanceIndex = {91, 65, 45, 36, 66, 61, 63, 42, 61};

    // Method to perform linear regression
    public static double[] linearRegression(double[] x, double[] y) {
        int n = x.length;
        double sumX = Arrays.stream(x).sum();
        double sumY = Arrays.stream(y).sum();
        double sumX2 = Arrays.stream(x).map(i -> i * i).sum();
        double sumXY = 0;
        for (int i = 0; i < n; i++) {
            sumXY += x[i] * y[i];
        }
        double b = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double a = (sumY - b * sumX) / n;
        return new double[]{a, b};
    }

    // Method to perform power regression
    public static double[] powerRegression(double[] x, double[] y) {
        double[] logX = Arrays.stream(x).map(Math::log).toArray();
        double[] logY = Arrays.stream(y).map(Math::log).toArray();
        double[] logCoeff = linearRegression(logX, logY);
        double a = Math.exp(logCoeff[0]);
        double b = logCoeff[1];
        return new double[]{a, b};
    }

    // Method to calculate RMS error
    public static double calculateRMSError(double[] x, double[] y, double[] coeff, boolean isPower) {
        double sumSquaredError = 0;
        for (int i = 0; i < x.length; i++) {
            double predictedY = isPower ? coeff[0] * Math.pow(x[i], coeff[1]) : coeff[0] + coeff[1] * x[i];
            sumSquaredError += Math.pow(predictedY - y[i], 2);
        }
        return Math.sqrt(sumSquaredError / x.length);
    }

    // Main method for testing
    public static void main(String[] args) {
        // Linear regression
        double[] linearCoeff = linearRegression(hoursStudied, performanceIndex);
        System.out.println("Linear Regression Coefficients: a = " + linearCoeff[0] + ", b = " + linearCoeff[1]);
        double linearRMS = calculateRMSError(hoursStudied, performanceIndex, linearCoeff, false);
        System.out.println("Linear Regression RMS Error: " + linearRMS);

        // Power regression
        double[] powerCoeff = powerRegression(hoursStudied, performanceIndex);
        System.out.println("Power Regression Coefficients: a = " + powerCoeff[0] + ", b = " + powerCoeff[1]);
        double powerRMS = calculateRMSError(hoursStudied, performanceIndex, powerCoeff, true);
        System.out.println("Power Regression RMS Error: " + powerRMS);

        // Print the predicted values for both models
        System.out.println("\nPredicted Performance Index:");
        System.out.println("Hours Studied | Linear Prediction | Power Prediction");
        for (int i = 0; i < hoursStudied.length; i++) {
            double linearPrediction = linearCoeff[0] + linearCoeff[1] * hoursStudied[i];
            double powerPrediction = powerCoeff[0] * Math.pow(hoursStudied[i], powerCoeff[1]);
            System.out.printf("%13.1f | %16.2f | %15.2f\n", hoursStudied[i], linearPrediction, powerPrediction);
        }
    }
}
