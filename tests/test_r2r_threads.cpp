#include "notorious_fft.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <cmath>

int main() {
    std::atomic<int> failures(0);
    std::vector<std::thread> workers;
    for (int thread = 0; thread < 4; ++thread) workers.emplace_back([thread, &failures]() {
        const int n0 = 3+thread, n1 = 4, n2 = 5;
        const size_t count = (size_t)n0*n1*n2;
        std::vector<double> x(count), y(count), original(count);
        for (size_t i = 0; i < count; ++i) original[i] = std::sin((double)i+thread*0.7);
        auto *f = notorious_fft_plan_r2r_3d(n0,n1,n2,x.data(),y.data(),
            NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_REDFT10,NOTORIOUS_FFT_REDFT10,0);
        auto *b = notorious_fft_plan_r2r_3d(n0,n1,n2,y.data(),y.data(),
            NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT01,NOTORIOUS_FFT_REDFT01,0);
        if (!f || !b) ++failures;
        else for (int rep = 0; rep < 20; ++rep) {
            x = original;
            notorious_fft_execute_r2r(f,x.data(),y.data());
            notorious_fft_execute_r2r(b,y.data(),y.data());
            for (size_t i = 0; i < count; ++i)
                if (std::abs(y[i]/(8.0*count)-original[i]) > 1e-11) ++failures;
        }
        notorious_fft_destroy_io_plan(f); notorious_fft_destroy_io_plan(b);
    });
    for (auto &worker : workers) worker.join();
    return failures != 0;
}
