#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

void matrixMultiply(double** a, double** b, double** c, int size);

int main(int argc, char* argv[]) {
    if (argc > 1) {
        int size = atoi(argv[1]);

        double** a = new double* [size];
        double** b = new double* [size];
        double** c = new double* [size];

        for (int i = 0; i < size; ++i) {
            a[i] = new double[size];
            b[i] = new double[size];
            c[i] = new double[size];
        }

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i][j] = (double(rand()) / RAND_MAX);
                b[i][j] = (double(rand()) / RAND_MAX);
            }
        }

        double time_for_one_thread;
        double exec_start_time = omp_get_wtime();
        matrixMultiply(a, b, c, size);
        time_for_one_thread = omp_get_wtime() - exec_start_time;

        for (int num_threads = 1; num_threads <= 10; num_threads++) {
            double exec_start_time = omp_get_wtime();

#pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < 1; i++) {  // Execute the loop only once for each num_threads value
                matrixMultiply(a, b, c, size);
            }

            double exec_time = (omp_get_wtime() - exec_start_time);
            double efficiency = time_for_one_thread / exec_time;

            cout << "Number of threads: " << num_threads << "  Execution time (in seconds): " << exec_time << "  Efficiency: " << efficiency << endl;
        }

        for (int i = 0; i < size; ++i) {
            delete[] a[i];
            delete[] b[i];
            delete[] c[i];
        }
        delete[] a;
        delete[] b;
        delete[] c;
    } else {
        cout << "Please input the arguments" << endl;
    }
    return 0;
}

void matrixMultiply(double** a, double** b, double** c, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}