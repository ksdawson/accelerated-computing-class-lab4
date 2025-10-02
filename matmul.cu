#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

__device__ void load_tile(
    float const *src, uint32_t src_width, 
    float *dst, uint32_t dst_height, uint32_t dst_width
) {
    for (uint32_t idx = threadIdx.x; idx < dst_height * dst_width; idx += blockDim.x) {
        // Get index to copy
        uint32_t i = idx / dst_width;
        uint32_t j = idx % dst_width;
        // Copy mem over
        dst[i * dst_width + j] = src[i * src_width + j];
    }
    // Wait for all the memory to be copied
    __syncthreads();
}

__device__ void matmul_tile(
    uint32_t size_i, uint32_t size_j, uint32_t size_k, // Matrix dimensions
    float const *a, float const *b, float *c, // Matrices
    float *local_a, float *local_b, // Matrices in SRAM
    uint32_t n // Tile size
) {
    // Math: c_ik = a_i ⋅ b_k
    // Goal: c_ik in registers and a_i, b_k in L1 cache
    // Plan: Each thread works on one c_ik at a time

    // Each thread gets a c_ik
    uint32_t i = threadIdx.x / n;
    uint32_t k = threadIdx.x % n;

    // Keep c_ik in local register
    float local_c_ik = 0.0f;

    uint32_t a_height = n;
    uint32_t a_width = min(size_j, 390);
    uint32_t b_height = min(size_j, 390);
    uint32_t b_width = n;

    // Iterate over subtiles of a_i, b_k
    for (uint32_t idx = 0; idx < size_j; idx += a_width) {
        // Update a_width, b_height
        a_width = min(a_width, size_j - idx);
        b_height = min(b_height, size_j - idx);

        // Load subtiles
        load_tile(
            a, size_j,
            local_a, a_height, a_width
        );
        load_tile(
            b, size_k,
            local_b, b_height, b_width
        );

        // Iterate over a_i, b_k
        for (uint32_t j = 0; j < a_width; ++j) {
            local_c_ik += local_a[i * a_width + j] * local_b[j * b_width + k];
        }

        // Move buffers
        a += a_width;
        b += b_height * size_k;

        __syncthreads();
    }

    // Write back to main memory at the end
    c[i * size_k + k] = local_c_ik;
}

__global__ void matmul_l1(
    int32_t size_i, int32_t size_j, int32_t size_k,
    float const *a, float const *b, float *c
) {
    // c_ik tiles are 32x32
    constexpr uint32_t n = 32;

    // Grid dimensions
    uint32_t tiles_per_row = size_k / n;
    uint32_t tiles_per_col = size_j / n;

    // Setup the block's SRAM
    extern __shared__ float sram[];
    float *local_a = sram;
    float *local_b = sram + 32 * min(size_j, 390); // 25000/(2*32) = 390

    // Iterate over tiles
    for (uint32_t idx = blockIdx.x; idx < tiles_per_col * tiles_per_row; idx += gridDim.x) {
        // Tile indices
        // uint32_t tile_i = idx / tiles_per_row; // Row major
        // uint32_t tile_k = idx % tiles_per_row;
        uint32_t tile_k = idx / tiles_per_col; // Col major
        uint32_t tile_i = idx % tiles_per_col;

        // Move buffers
        float const *tile_a = a + tile_i * n * size_j;
        float const *tile_b = b + tile_k * n;
        float *tile_c = c + tile_i * n * size_j + tile_k * n;

        matmul_tile(
            size_i, size_j, size_k,
            tile_a, tile_b, tile_c,
            local_a, local_b,
            n
        );
    }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    // Setup the block SRAM
    int shmem_size_bytes = 100 * 1013; // Max 100 KB per block
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes
    ));
    matmul_l1<<<48, 32 * 32, shmem_size_bytes>>>(size_i, size_j, size_k, a, b, c);
}

// Part 2 lower bound: 17ms

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
    // Lower bound: 5ms
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
