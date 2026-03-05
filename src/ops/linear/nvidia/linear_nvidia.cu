#include "llaisys.h"
#include "../../ops.hpp"
#include "../../../device/nvidia/utils.cuh"


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LDST128BITS(value) ((reinterpret_cast<float4*>(&(value))[0]))
#define FLOAT(value) ((reinterpret_cast<float*>(&(value))[0]))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile( \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"\
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) \
        : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile(\
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"\
        : "=r"(R0), "=r"(R1) \
        : "r"(addr))
#define HMMA16816F32(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3) \
    asm volatile(\
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "\
        "{%8, %9}, {%10, %11, %12, %13};\n"\
        : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3) \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))
#define CP_ASYNC_CG(dst, src, bytes)       \
    asm volatile(                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_WAIT_GROUP(n)  \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

namespace {

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16, 
const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
const int WARP_TILE_K = 2, const int A_PAD = 8, const int B_PAD = 8,
const int K_STAGE = 3>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_kernel(
        const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, const __nv_bfloat16 *__restrict__ bias,
        __nv_bfloat16 *__restrict__ C, int M, int N, int K
    ) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = CEIL(K, MMA_K * WARP_TILE_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    constexpr int BK = MMA_K;
    
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16 *s_a = smem;
    __nv_bfloat16 *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BN * (BK + B_PAD);
    constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
    constexpr int s_b_mma_k_store_offset = K_STAGE * BN * (BK + B_PAD);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = tid % 2 * 8;
    int load_smem_b_n = tid / 2;
    int load_smem_b_k = tid % 2 * 8;
    
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    //if(load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][4];
    for(int i = 0;i < WARP_TILE_M;++i) {
        for(int j = 0;j < WARP_TILE_N;++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    for(int k = 0;k < (K_STAGE - 1);++k) {
        int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;
        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(__nv_bfloat16));
        if(load_gmem_a_m < M && load_gmem_a_k + 8 <= K) {
            CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_a[(k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k + i)] = (load_gmem_a_m < M && load_gmem_a_k + i < K) ? A[load_gmem_a_addr + i] : __nv_bfloat16(0);
            }
        }
        
        uint32_t load_smem_a_mma_k_ptr = load_smem_a_ptr + s_a_mma_k_store_offset * sizeof(__nv_bfloat16);
        if(load_gmem_a_m < M && load_gmem_a_k + 16 + 8 <= K) {
            CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_a[(s_a_mma_k_store_offset + k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k + i)] = (load_gmem_a_m < M && load_gmem_a_k + 16 + i < K) ? A[load_gmem_a_addr + 16 + i] : __nv_bfloat16(0);
            }
        }
        
        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) * sizeof(__nv_bfloat16));
        if(load_gmem_b_n < N && load_gmem_b_k + 8 <= K) {
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_b[(k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) + i] = (load_gmem_b_n < N && load_gmem_b_k + i < K) ? B[load_gmem_b_addr + i] : __nv_bfloat16(0);
            }
        }
        uint32_t load_smem_b_mma_k_ptr = load_smem_b_ptr + s_b_mma_k_store_offset * sizeof(__nv_bfloat16);
        if(load_gmem_b_n < N && load_gmem_b_k + 16 + 8 <= K) {
            CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_b[(k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k + s_b_mma_k_store_offset) + i] = (load_gmem_b_n < N && load_gmem_b_k + 16 + i < K) ? B[load_gmem_b_addr + 16 + i] : __nv_bfloat16(0);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
    uint32_t RA[2][WARP_TILE_M][4];
    uint32_t RB[2][WARP_TILE_N][2];
    int reg_store_idx = 0;
    int reg_load_idx = 1;
    {
        for(int i = 0;i < WARP_TILE_M;++i) {
            int warp_smem_a_m = warp_m * WARP_TILE_M * MMA_M + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = lane_id / 16 * 8;
            uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_smem_a_ptr);
        }

        for(int i = 0;i < WARP_TILE_N;++i) {
            int warp_smem_b_n = warp_n * WARP_TILE_N * MMA_N + i * MMA_N;
            int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
            int lane_smem_b_k = lane_id / 8 * 8;
            uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X2(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1], lane_smem_b_ptr);
        }
    }
    
    for(int k = (K_STAGE - 1);k < NUM_K_TILES;++k) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
        int smem_sel = (k + 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;

        int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(__nv_bfloat16));
        if(load_gmem_a_m < M && load_gmem_a_k + 8 <= K) {
            CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_a[(smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) + i] = (load_gmem_a_m < M && load_gmem_a_k + i < K) ? A[load_gmem_a_addr + i] : __nv_bfloat16(0);
            }
        }
        uint32_t load_smem_a_mma_k_ptr = (load_smem_a_ptr + s_a_mma_k_store_offset * sizeof(__nv_bfloat16));
        if(load_gmem_a_m < M && load_gmem_a_k + 16 + 8 <= K) {
            CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_a[(smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k + s_a_mma_k_store_offset) + i] = (load_gmem_a_m < M && load_gmem_a_k + 16 + i < K) ? A[load_gmem_a_addr + 16 + i] : __nv_bfloat16(0);
            }
        }
        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) * sizeof(__nv_bfloat16));
        if(load_gmem_b_n < N && load_gmem_b_k + 8 <= K) {
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_b[(smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) + i] = (load_gmem_b_n < N && load_gmem_b_k + i < K) ? B[load_gmem_b_addr + i] : __nv_bfloat16(0);
            }
        }
        uint32_t load_smem_b_mma_k_ptr = (load_smem_b_ptr + s_b_mma_k_store_offset * sizeof(__nv_bfloat16));
        if(load_gmem_b_n < N && load_gmem_b_k + 16 + 8 <= K) {
            CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr + 16], 16);
        } else {
            for(int i = 0;i < 8;++i) {
                s_b[(smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k + s_b_mma_k_store_offset) + i] = (load_gmem_b_n < N && load_gmem_b_k + 16 + i < K) ? B[load_gmem_b_addr + 16 + i] : __nv_bfloat16(0);
            }
        }
        
        CP_ASYNC_COMMIT_GROUP();
        for(int i = 0;i < WARP_TILE_M;++i) {
            int warp_smem_a_m = warp_m * WARP_TILE_M * MMA_M + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = lane_id / 16 * 8;
            uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (reg_store_idx * s_a_mma_k_store_offset + smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_smem_a_ptr);
        }
        for(int i = 0;i < WARP_TILE_N;++i) {
            int warp_smem_b_n = warp_n * WARP_TILE_N * MMA_N + i * MMA_N;
            int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
            int lane_smem_b_k = lane_id / 8 * 8;
            uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (reg_store_idx * s_b_mma_k_store_offset + smem_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X2(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1], lane_smem_b_ptr);
        }
        for(int i = 0;i < WARP_TILE_M;++i) {
            for(int j = 0;j < WARP_TILE_N;++j) {
                HMMA16816F32(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
            }
        }
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
        for(int i = 0;i < WARP_TILE_M;++i) {
            for(int j = 0;j < WARP_TILE_N;++j) {
                HMMA16816F32(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                        RB[reg_load_idx][j][0], RB[reg_load_idx][j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
        int reg_smem_sel = (smem_sel + 1) % K_STAGE;
        for(int i = 0;i < WARP_TILE_M;++i) {
            int warp_smem_a_m = warp_m * WARP_TILE_M * MMA_M + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = lane_id / 16 * 8;
            uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (reg_smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_smem_a_ptr);
        }
        for(int i = 0;i < WARP_TILE_N;++i) {
            int warp_smem_b_n = warp_n * WARP_TILE_N * MMA_N + i * MMA_N;
            int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
            int lane_smem_b_k = lane_id / 8 * 8;
            uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (reg_smem_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) * sizeof(__nv_bfloat16));
            LDMATRIX_X2(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1], lane_smem_b_ptr);
        }
    }
    if constexpr ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
    {
         for(int k = 0;k < K_STAGE - 1;++k) {
            reg_load_idx ^= 1;
            reg_store_idx ^= 1;
            int smem_sel = (NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE;
            for(int i = 0;i < WARP_TILE_M;++i) {
                int warp_smem_a_m = warp_m * WARP_TILE_M * MMA_M + i * MMA_M;
                int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
                int lane_smem_a_k = lane_id / 16 * 8;
                uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (reg_store_idx * s_a_mma_k_store_offset + smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(__nv_bfloat16));
                LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_smem_a_ptr);
            }
            for(int i = 0;i < WARP_TILE_N;++i) {
                int warp_smem_b_n = warp_n * WARP_TILE_N * MMA_N + i * MMA_N;
                int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
                int lane_smem_b_k = lane_id / 8 * 8;
                uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (reg_store_idx * s_b_mma_k_store_offset + smem_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) * sizeof(__nv_bfloat16));
                LDMATRIX_X2(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1], lane_smem_b_ptr);
            }
            for(int i = 0;i < WARP_TILE_M;++i) {
                for(int j = 0;j < WARP_TILE_N;++j) {
                    HMMA16816F32(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[0][i][0], RA[0][i][1], RA[0][i][2], RA[0][i][3], RB[0][j][0], RB[0][j][1], 
                            RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
            for(int i = 0;i < WARP_TILE_M;++i) {
                for(int j = 0;j < WARP_TILE_N;++j) {
                    HMMA16816F32(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[1][i][0], RA[1][i][1], RA[1][i][2], RA[1][i][3], RB[1][j][0], RB[1][j][1], 
                            RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
            reg_store_idx ^= 1;
            int reg_stage_sel = (smem_sel + 1) % K_STAGE;
            for(int i = 0;i < WARP_TILE_M;++i) {
                int warp_smem_a_m = warp_m * WARP_TILE_M * MMA_M + i * MMA_M;
                int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
                int lane_smem_a_k = lane_id / 16 * 8;
                uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (reg_store_idx * s_a_mma_k_store_offset + reg_stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(__nv_bfloat16));
                LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_smem_a_ptr);
            }
            for(int i = 0;i < WARP_TILE_N;++i) {
                int warp_smem_b_n = warp_n * WARP_TILE_N * MMA_N + i * MMA_N;
                int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
                int lane_smem_b_k = lane_id / 8 * 8;
                uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (reg_store_idx * s_b_mma_k_store_offset + reg_stage_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) * sizeof(__nv_bfloat16));
                LDMATRIX_X2(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1], lane_smem_b_ptr);
            }
         }
    }
    for(int i = 0;i < WARP_TILE_M;++i) {
        __nv_bfloat16 RC16[2][WARP_TILE_N][8];
        for(int j = 0;j < WARP_TILE_N;++j) {
            int store_gmem_c_n = bx * BN + warp_n * WARP_TILE_N * MMA_N + j * MMA_N + lane_id % 4 * 2;
            //printf("%.4f\n", FLOAT(RC[i][j][0]));
            RC16[0][j][0] = __float2bfloat16(FLOAT(RC[i][j][0]));
            RC16[0][j][1] = __float2bfloat16(FLOAT(RC[i][j][1]));
            RC16[1][j][0] = __float2bfloat16(FLOAT(RC[i][j][2]));
            RC16[1][j][1] = __float2bfloat16(FLOAT(RC[i][j][3]));
            if(store_gmem_c_n < N) {
                RC16[0][j][0] += bias[store_gmem_c_n];
                RC16[1][j][0] += bias[store_gmem_c_n];
            }
            
            if(store_gmem_c_n + 1 < N) {
                RC16[0][j][1] += bias[store_gmem_c_n + 1];
                RC16[1][j][1] += bias[store_gmem_c_n + 1];
            }
        }
        for(int j = 0;j < WARP_TILE_N;++j) {
            RC16[0][j][2] = __shfl_sync(0xffffffff, RC16[0][j][0], lane_id + 1);
            RC16[0][j][3] = __shfl_sync(0xffffffff, RC16[0][j][1], lane_id + 1);
            RC16[0][j][4] = __shfl_sync(0xffffffff, RC16[0][j][0], lane_id + 2);
            RC16[0][j][5] = __shfl_sync(0xffffffff, RC16[0][j][1], lane_id + 2);
            RC16[0][j][6] = __shfl_sync(0xffffffff, RC16[0][j][0], lane_id + 3);
            RC16[0][j][7] = __shfl_sync(0xffffffff, RC16[0][j][1], lane_id + 3);

            RC16[1][j][2] = __shfl_sync(0xffffffff, RC16[1][j][0], lane_id + 1);
            RC16[1][j][3] = __shfl_sync(0xffffffff, RC16[1][j][1], lane_id + 1);
            RC16[1][j][4] = __shfl_sync(0xffffffff, RC16[1][j][0], lane_id + 2);
            RC16[1][j][5] = __shfl_sync(0xffffffff, RC16[1][j][1], lane_id + 2);
            RC16[1][j][6] = __shfl_sync(0xffffffff, RC16[1][j][0], lane_id + 3);
            RC16[1][j][7] = __shfl_sync(0xffffffff, RC16[1][j][1], lane_id + 3);
            
            if(lane_id % 4 == 0) {
                int store_gmem_c_m = by * BM + warp_m * WARP_TILE_M * MMA_M + i * MMA_M + lane_id / 4;
                int store_gmem_c_n = bx * BN + warp_n * WARP_TILE_N * MMA_N + j * MMA_N;
                if(store_gmem_c_m < M && store_gmem_c_n + 8 <= N)
                    LDST128BITS(C[store_gmem_c_m * N + store_gmem_c_n]) = LDST128BITS(RC16[0][j][0]);
                else if(store_gmem_c_m < M) {
                    for(int k = 0;store_gmem_c_n + k < N;++k) {
                        C[store_gmem_c_m * N + store_gmem_c_n + k] = RC16[0][j][k];
                    }
                }
                if(store_gmem_c_m + 8 < M && store_gmem_c_n + 8 <= N)
                    LDST128BITS(C[(store_gmem_c_m + 8) * N + store_gmem_c_n]) = LDST128BITS(RC16[1][j][0]);
                else if(store_gmem_c_m + 8 < M) {
                    for(int k = 0;store_gmem_c_n + k < N;++k) {
                        C[(store_gmem_c_m + 8) * N + store_gmem_c_n + k] = RC16[1][j][k];
                    }
                }
            }
        }
    }
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for(int i = WARP_SIZE >> 1;i > 0;i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}
__global__ void hgemv_k128_f16_kernel(const __nv_bfloat16 *__restrict__ A, const __nv_bfloat16 *__restrict__ B, const __nv_bfloat16 *__restrict__ bias, __nv_bfloat16 *__restrict__ C, int M, int N, int K) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int tx = tid % WARP_SIZE;
    const int ty = tid / WARP_SIZE;
    const int n = bid * 8 + ty;
    if(n >= N) return;
    const int NUM_WARPS = CEIL(K, 128);
    float sum = 0;
    for(int i = 0;i < NUM_WARPS;++i) {
        int load_gmem_a_k = i * 128 + tx * 4;
        float a_0 = __bfloat162float(A[load_gmem_a_k]);
        float a_1 = __bfloat162float(A[load_gmem_a_k + 1]);
        float a_2 = __bfloat162float(A[load_gmem_a_k + 2]);
        float a_3 = __bfloat162float(A[load_gmem_a_k + 3]);
        float b_0 = __bfloat162float(B[n * K + load_gmem_a_k]);
        float b_1 = __bfloat162float(B[n * K + load_gmem_a_k + 1]);
        float b_2 = __bfloat162float(B[n * K + load_gmem_a_k + 2]);
        float b_3 = __bfloat162float(B[n * K + load_gmem_a_k + 3]);
        sum += a_0 * b_0 + a_1 * b_1 + a_2 * b_2 + a_3 * b_3;
    }
    sum = warp_reduce_sum(sum);
    if(tx == 0) {
        C[n] = __float2bfloat16(sum) + bias[n];
    }
}

template<typename T>
__global__ void add_bias(T* out, const T* in, const T* bias, size_t M, size_t N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= M * N) return ;
    int col = idx % N;
    out[idx] = in[idx] + bias[col];
}
template <typename T, cudaDataType_t CType>
void linear_(std::byte *out, // [m, n]  (row major)  /  [n, m] (col major)
             const std::byte *in, // [m, k] (row major) / [k, m] (col major)
             const std::byte *weight, // [n, k] (row major) / [k, n] (col major)
             const std::byte *bias,
             size_t m,  // batch (rows of out)
             size_t k,  // in_dim
             size_t n,  // out_dim (cols of out)
             cudaStream_t stream) {
    //__nv_bfloat16
    if(std::is_same_v<T, __nv_bfloat16> && bias != nullptr) {
        if(m == 1) {
            dim3 block(256);
            dim3 grid(CEIL(n, 8));
            hgemv_k128_f16_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<const __nv_bfloat16*>(weight), 
            reinterpret_cast<const __nv_bfloat16*>(bias),reinterpret_cast<__nv_bfloat16*>(out), m, n, k);
            return;
        }
        // MMA kernel 要求 bias 非空，否则会空指针解引用导致非法访存
        const int MMA_M = 16;
        const int MMA_N = 8;
        const int MMA_K = 16;
        const int MMA_TILE_M = 2;
        const int MMA_TILE_N = 4; 
        const int WARP_TILE_M = 4;
        const int WARP_TILE_N = 4;
        const int WARP_TILE_K = 2;
        const int A_PAD = 8;
        const int B_PAD = 8;
        const int K_STAGE = 4;
        int sram_size = (2 * K_STAGE * MMA_TILE_M * WARP_TILE_M * MMA_M * (MMA_K + A_PAD) + 2 * K_STAGE * (MMA_TILE_N * WARP_TILE_N * MMA_N) * (MMA_K + B_PAD)) * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(                                                      
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_kernel<               
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, K_STAGE>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        dim3 block(256);
        dim3 grid(CEIL(n, 128), CEIL(m, 128));
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_kernel<               
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, K_STAGE><<<grid, block, sram_size, stream>>>(reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<const __nv_bfloat16*>(weight),
                                                    reinterpret_cast<const __nv_bfloat16*>(bias), reinterpret_cast<__nv_bfloat16*>(out), m, n, k);
    } else {
        auto &runtime = llaisys::core::context().runtime();
        auto handle = runtime.cublasHandle();

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        
        const auto algo = std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        // 数学目标（row 语义，不涉及存储）：
        //   out(m,n) = in(m,k) * weight^T(k,n)
        //
        // 关键等价（不拷贝、不物理转置）：
        //   (in * weight^T)^T = weight * in^T
        //
        // 存储等价（字节层面）：
        //   row(m,n) <=> col(n,m)
        //
        // 于是我们在 cuBLAS（列主序）里让它计算：
        //   C(n,m) = weight(n,k) * in^T(k,m) = out^T(n,m)
        //
        // 指针与参数映射（只改变“解释方式”，不做物理转置）：
        //   A <- weight  // row: [n,k]  → cuBLAS(col视角): [k,n],  lda = k,  opA = T  → (n,k)
        //   B <- in      // row: [m,k]  → cuBLAS(col视角): [k,m],  ldb = k,  opB = N  → (k,m) = in^T
        //   C <- out     // row: [m,n]  → cuBLAS(col视角): [n,m],  ldc = n
        //
        //   维度：m' = n,  n' = m,  k' = k
        //
        // opA/opB 是“数学转置标志”，用于纠正 cuBLAS 的列主序解释；不进行任何物理数据转置/拷贝。
        auto compute_type = std::is_same_v<T, __half> ? CUBLAS_COMPUTE_32F_FAST_16F : 
                        std::is_same_v<T, __nv_bfloat16> ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;



        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,                 // opA, opB
            static_cast<int>(n),                      // m' = n
            static_cast<int>(m),                      // n' = m
            static_cast<int>(k),                      // k
            &alpha,
            reinterpret_cast<const T *>(weight), CType, static_cast<int>(k), // A: lda = k
            reinterpret_cast<const T *>(in),     CType, static_cast<int>(k), // B: ldb = k
            &beta,
            reinterpret_cast<T *>(out),          CType, static_cast<int>(n), // C: ldc = n
            compute_type,
            algo);
        
        if(bias) {
            dim3 block(256);
            dim3 grid(CEIL(m * n, 256));
            add_bias<T><<<grid, block, 0, stream>>>(reinterpret_cast<T*>(out), reinterpret_cast<const T*>(out), reinterpret_cast<const T*>(bias), m, n);
        }
    }
}


} // anomynous namespace


namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            size_t m, size_t k, size_t n,
            llaisysDataType_t type) {

    auto s = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
		case LLAISYS_DTYPE_F32:
			linear_<float, CUDA_R_32F>(out, in, weight, bias,
									   m, k, n, s);
			break;
		case LLAISYS_DTYPE_F16:
			linear_<__half, CUDA_R_16F>(out, in, weight, bias,
										m, k, n, s);
			break;
		case LLAISYS_DTYPE_BF16:
			linear_<__nv_bfloat16, CUDA_R_16BF>(out, in, weight, bias,
												m, k, n, s);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia