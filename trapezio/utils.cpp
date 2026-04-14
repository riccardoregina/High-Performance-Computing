#include "utils.h"
#include <chrono>

double get_cur_time() {
    // Ottiene il tempo corrente ad alta risoluzione
    auto now = std::chrono::high_resolution_clock::now();
    
    // Converte il tempo in secondi sotto forma di double
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

float float_avg(const float* v, const int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += v[i];
    }
    return sum / size;
}

double double_avg(const double* v, const int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += v[i];
    }
    return sum / size;
}
