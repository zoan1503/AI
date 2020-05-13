package AI;

import java.util.Random;

public class Perceptron {
    double[] w;
    double bias;

    public void train(double[][] x, double[] y, double learning_rate, int iter) {
        int n = x[0].length;
        int p = y.length;
        w = new double[n];
        Random r = new Random();

        bias = r.nextDouble();

        for (int i = 0; i < n; i ++) {
            w[i] = r.nextDouble();
        }

        double totalE;
        double error;
        int temp = 0;

        do {
            totalE = 0.0;
            for (int i = 0; i < p; i ++) {
                error = y[i] - z(x[i]);

                for (int j = 0; j < n; j ++) {
                    w[j] += error * learning_rate * x[i][j];
                }
                bias += learning_rate * error;
                totalE += Math.abs(error);
            }
            temp ++;
        } while (temp < iter && totalE != 0);
    }

    public double z(double[] x) {
        double z = 0.0;
        for (int i = 0; i < x.length; i ++) {
            z += w[i]*x[i];
        }
        return sigmoid(z + bias);
    }

    public double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void print() {
        System.out.println(w[0] + ".x1 + " + w[1] + ".x2 + " + bias + " = y");
    }

    public void print_(double[][] x) {
        int n = x.length;
        for (int i = 0; i < n; i ++) {
            System.out.println("(" + x[i][0] + ", " + x[i][1] + ") ---> " + sigmoid(w[0] * x[i][0] + w[1] * x[i][1] + bias));
        }
    }

    public double think(double[][] a) {
        int n = a.length;
        double y = 0.0;
        for (int i = 0; i < n; i ++) {
            y = w[0] * a[i][0] + w[1] * a[i][1] + bias;
        }
        return y;
    }

    public static void main(String[] args) {
        //double[][] x = {{10,1}, {5,2}, {6,1.8}, {7,1}, {8,2}, {9,0.5}, {4,3}, {5,2.5}, {8,1}, {4,2.5}, {8,0.1}, {7,0.15}, {4,1}, {5,0.8}, {7,0.3}, {4,1}, {5,0.5}, {6,0.3}, {7,0.2}, {8,0.15}};
        //double[] y = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double[][] x = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        };
        double[] y_and = {0, 0, 0, 1};
        double[] y_or = {0, 1, 1, 1};
        //double[] y_xor = {0, 1, 1, 0};

        Perceptron test = new Perceptron();
        test.train(x, y_and, 0.01, 100000);
        System.out.println("AND:");
        test.print_(x);

        test.train(x, y_or, 0.01, 100000);
        System.out.println("OR:");
        test.print_(x);

        //test.train(x, y_xor, 0.01, 100000);
        //System.out.println("XOR:");
        //test.print_(x);

        //test
        //double[][] a = {{7, 0.72}};
        //double res = test.think(a);
        //if (res > 0)
        //    System.out.println("1");
        //else
        //    System.out.println("0");
    }
}
