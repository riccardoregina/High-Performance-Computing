#include "utils.h"
#include <chrono>

double get_cur_time() {
    // Ottiene il tempo corrente ad alta risoluzione
    auto now = std::chrono::high_resolution_clock::now();
    
    // Converte il tempo in secondi sotto forma di double
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}