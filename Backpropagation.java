package AI;

import java.util.Random;

public class Backpropagation {
    double[][] w;
    double bias;
    double[] net;
    double[] h_w;
    double h_bias;
    double[] net_h;
    double[] out_h;
    double[] calculated_out;
    Random r = new Random();

    public void train(double[][] x, double[] y, int m, double learning_rate, double iter){
        h_w = new double[m];
        net_h = new double[m];
        out_h = new double[m];
        int n = x[0].length;
        int p = y.length;
        net = new double[p];
        calculated_out = new double[p];
        w = new double[m][n];

        for (int i = 0; i < m; i ++) {
            for (int j = 0; j < n; j ++) {
                w[i][j] = r.nextDouble() * 1.0;
            }
        }
        for (int i = 0; i < m; i ++) {
            h_w[i] = r.nextDouble() * 1.0;
        }
        bias = r.nextDouble();
        h_bias = r.nextDouble();
        int temp = 0;
        double error;
        double total_error;
        double delta;

        do {
            total_error = 0.0;
            for (int i = 0; i < p; i ++) {
                delta = 0.0;
                net[i] = h_bias;
                for (int j = 0; j < m; j ++) {
                    net_h[j] = bias;
                    for (int k = 0; k < n; k ++) {
                        net_h[j] += w[j][k] * x[i][k];
                    }
                    out_h[j] = sigmoid(net_h[j]);
                    net[i] += out_h[j] * h_w[j];
                }
                calculated_out[i] = sigmoid(net[i]);
                error = 0.5 * Math.pow(y[i] - calculated_out[i], 2);
                delta = (calculated_out[i] - y[i]) * calculated_out[i] * (1 - calculated_out[i]);
                for (int j = 0; j < m; j ++) {
                    for (int k = 0; k < n; k ++) {
                        w[j][k] -= delta * h_w[j] * out_h[j] * (1 - out_h[j]) * x[i][k] * learning_rate;
                    }
                }
                for (int j = 0; j < m; j ++) {
                    bias -= delta * h_w[j] * out_h[j] * (1 - out_h[j]) * learning_rate;
                    h_w[j] -= delta * out_h[j] * learning_rate;
                }
                h_bias -= delta * learning_rate;
                total_error += Math.abs(error);
            }
            temp ++;
        } while (temp < iter && total_error != 0);

        //System.out.println(total_error);

        for (int i = 0; i < p; i ++) {
            System.out.println("(" + x[i][0] + ", " + x[i][1] + ") --> " + calculated_out[i]);
        }
    }

    public double sigmoid(double x) {
        return 1/ (1 + Math.exp(-x));
    }

    public static void main(String[] args) {
        double[][] x = {
                {1, 1},
                {1, 0},
                {0, 1},
                {0, 0},
        };
        double[] y_xor = {0, 1, 1, 0};
        double[] y_or = {1, 1, 1, 0};

        Backpropagation test = new Backpropagation();
        //train XOR
        System.out.println("XOR:");
        test.train(x, y_xor, 3, 0.5, 100000);
        //train OR
        //System.out.println("OR:");
        //
        //test.train(x, y_or, 3, 0.5, 100000);
    }
}
