#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include "CoMDTypes.h"
#include "gpu_types.h"
#include <cuda_runtime.h>


EXTERN_C void ljForceGpu(SimGpu sim);

EXTERN_C void updateNeighborsGpu(SimGpu sim, int * temp);
EXTERN_C void updateNeighborsGpuAsync(SimGpu sim, int * temp, int nCells, int * cellList, cudaStream_t stream);
EXTERN_C void eamForce1Gpu(SimGpu sim, int method);
EXTERN_C void eamForce2Gpu(SimGpu sim, int method);
EXTERN_C void eamForce3Gpu(SimGpu sim, int method);

// latency hiding opt
EXTERN_C void eamForce1GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream);
EXTERN_C void eamForce2GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream);
EXTERN_C void eamForce3GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, cudaStream_t stream);


EXTERN_C void emptyNeighborListGpu(SimGpu * sim, int boundaryFlag);

EXTERN_C int compactCellsGpu(char* work_d, int nCells, int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets, int * d_workScan, real3_old shift, cudaStream_t stream);
EXTERN_C void unloadAtomsBufferToGpu(char *buf, int nBuf, SimFlat *s, char *gpu_buf, cudaStream_t stream);
EXTERN_C void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);
EXTERN_C void unloadForceBufferToGpu(char *buf, int nBuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf, cudaStream_t stream);

EXTERN_C void initHashTableGpu(HashTableGpu* hashTable, int nMaxEntries);

EXTERN_C void getAtomMsgSoAPtr(char* const buffer, AtomMsgSoA *atomMsg, int n);
#endif
