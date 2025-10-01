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

// Helper to load/store data
__device__ void copy_mem(float *src, float *dst, uint32_t buffer_width,
    uint32_t n_rows, uint32_t start_i
    ) {
    // Copy data from one buffer to another
    for (uint32_t thread_idx = threadIdx.x; thread_idx < n_rows * buffer_width; thread_idx += blockDim.x) {
        // Get local idx
        uint32_t i = thread_idx / buffer_width;
        uint32_t j = thread_idx % buffer_width;
        // Get dst idx
        uint32_t dst_idx = i * buffer_width + j;
        // Get src idx
        uint32_t src_idx = (start_i + i) * buffer_width + j;
        // Copy memory over
        dst[dst_idx] = src[src_idx];
    }
    __syncthreads();
}

__device__ void copy_mem_col(float *src, float *dst, uint32_t buffer_width,
    uint32_t n_cols, uint32_t start_j
    ) {
    // Copy data from one buffer to another
    for (uint32_t thread_idx = threadIdx.x; thread_idx < n_cols * buffer_width; thread_idx += blockDim.x) {
        // Get local idx
        uint32_t j = thread_idx / n_cols;
        uint32_t i = thread_idx % n_cols;
        // Get dst idx
        uint32_t dst_idx = i * buffer_width + j;
        // Get src idx
        uint32_t src_idx = i * buffer_width + (start_j + j);
        // Copy memory over
        dst[dst_idx] = src[src_idx];
    }
    __syncthreads();
}

__device__ void matmul_tile(
    uint32_t size_i, uint32_t size_j, uint32_t size_k, // Matrix dimensions
    float *a, float *b, float *c, // Matrices
    uint32_t start_i, uint32_t start_k, // Tile in c location
    uint32_t local_size_i, uint32_t local_size_k // Tile in c dimensions
    ) {
    // Math: c_ik = a_i ⋅ b_k
    // Goal: c_ik in registers and a_i, b_k in L1 cache
    // Plan: Each thread works on one c_ik at a time
    
    // More than one thread may need to split a c_ik
    // uint32_t threads_per_c_ik = max(blockDim.x / (local_size_i * local_size_k), 1);
    uint32_t threads_per_c_ik = 1;
    uint32_t start_idx = threadIdx.x / threads_per_c_ik;
    uint32_t end_idx = local_size_i * local_size_k;
    uint32_t step_size = blockDim.x / threads_per_c_ik;
    uint32_t local_size_j = size_j / threads_per_c_ik;
    
    // Iterate over c_ik's
    for (uint32_t idx = start_idx; idx < end_idx; idx += step_size) {
        // Get indices
        uint32_t i = idx / local_size_k;
        uint32_t k = idx % local_size_k;
        // Get local thread idx
        uint32_t local_idx = threadIdx.x % threads_per_c_ik;
        // Get j iteration bounds
        uint32_t start_j = local_size_j * local_idx;
        uint32_t end_j = start_j + local_size_j;
        // Keep c_ik in local register
        float local_c_ik = 0.0f;
        // Iterate over a_i, b_k
        for (uint32_t j = start_j; j < end_j; ++j) {
            local_c_ik += a[i * size_j + j] * b[j * local_size_k + k];
            // local_c_ik += a[i * size_j + j] * b[k * size_j + j];
        }
        // Write back to main memory at the end
        c[(start_i + i) * size_k + (start_k + k)] += local_c_ik; // Might cause race conditions
    }
}

__global__ void matmul_l1(
    int32_t size_i, int32_t size_j, int32_t size_k,
    float *a, float *b, float *c) {
    // Grid dimensions
    constexpr uint32_t tiles_per_row = 48; // Tuning parameter
    uint32_t tiles_per_col = gridDim.x / tiles_per_row;

    // Tile location
    uint32_t tile_i = blockIdx.x / tiles_per_row;
    uint32_t tile_k = blockIdx.x % tiles_per_row;

    // Tile dimensions
    uint32_t tile_height = size_i / tiles_per_col;
    uint32_t tile_width = size_k / tiles_per_row;

    // Handle extra
    uint32_t extra_height = size_i % tiles_per_col;
    uint32_t extra_width = size_k % tiles_per_row;
    // Spread extra over first rows and cols
    tile_height += (tile_i < extra_height) ? 1 : 0;
    tile_width += (tile_k < extra_width) ? 1 : 0;

    // Element location
    uint32_t start_i = tile_i < extra_height
        ? tile_i * tile_height
        : extra_height * (tile_height + 1) + (tile_i - extra_height) * tile_height;
    uint32_t start_k = tile_k < extra_width
        ? tile_k * tile_width
        : extra_width * (tile_width + 1) + (tile_k - extra_width) * tile_width;

    // Load as many b cols saving space for at least one a row
    uint32_t b_cols = min(tile_width, 25000 / size_j - 1);
    // Load as many a rows with the remaining space
    uint32_t a_rows = min(tile_height, 25000 / size_j - b_cols);

    // Setup the block's SRAM
    extern __shared__ float sram[];
    float *local_a = sram;
    float *local_b = sram + a_rows * size_j;

    // Iterate over the tile because the whole tile might not fit into the SRAM
    for (uint32_t k = 0; k < tile_width; k += b_cols) {
        // Load b cols
        uint32_t curr_b_cols = min(b_cols, tile_width - k);
        copy_mem_col(b, local_b, size_j, curr_b_cols, start_k + k);

        for (uint32_t i = 0; i < tile_height; i += a_rows) {
            // Load a rows
            uint32_t curr_a_rows = min(a_rows, tile_height - i);
            copy_mem(a, local_a, size_j, curr_a_rows, start_i + i);

            // Compute the subtile
            matmul_tile(
                size_i, size_j, size_k,
                local_a, local_b, c,
                start_i + i, start_k + k,
                curr_a_rows, curr_b_cols
            );

            // Need to wait for all threads to finish w/ what's in the SRAM
            __syncthreads();
        }
    }
}

__global__ void transpose_square_matrix(float *matrix, uint32_t n) {
    uint32_t tot_threads = gridDim.x * blockDim.x;
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = thread_index; idx < n * n; idx += tot_threads) {
        uint32_t i = idx / n;
        uint32_t j = idx % n;
        // Swap matrix[i][j] with matrix[j][i]
        if (i < j) {
            float temp = matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i];
            matrix[j * n + i] = temp;
        }
    }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    // We want: a -> row major, b -> col major, c -> row major
    // transpose_square_matrix<<<48, 32 * 32>>>(const_cast<float*>(b), size_j); // Separate kernel so synchronizes across the grid

    // Setup the block SRAM
    int shmem_size_bytes = 100 * 1013; // Max 100 KB per block
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes
    ));

    matmul_l1<<<48, 32 * 32, shmem_size_bytes>>>(size_i, size_j, size_k, const_cast<float*>(a), const_cast<float*>(b), c);
}
// Part 2 lower bound: 17ms
// Optimization strategy:
// (1) a, b in L1 cache; c in registers
// (2) prefetch next tile into L2
// (3) transpose b

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
