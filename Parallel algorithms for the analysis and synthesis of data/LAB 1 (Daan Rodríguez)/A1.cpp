#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Uso: " << argv[0] << "tamaño_vector" << std::endl;
        return 1;
    }

    int tama_vector = atoi(argv[1]);
    vector<int> vec(tama_vector);

    srand(time(nullptr));
    for (int i = 0; i < tama_vector; i++) {
        vec[i] = rand();
    }

    double inicio, fin;

    for (int num_hilos = 1; num_hilos <= 10; num_hilos++) {
        omp_set_num_threads(num_hilos);
        int valor_max = 0;  

        inicio = omp_get_wtime();

        #pragma omp parallel for reduction(max:valor_max)
        for (int i = 0; i < tama_vector; i++) {
            if (vec[i] > valor_max) {
                valor_max = vec[i];
            }
        }

        fin = omp_get_wtime();

        cout << "Number of threads: " << num_hilos << endl;
        cout << "Execution time: " << fin - inicio << endl;
    }

    return 0;
}