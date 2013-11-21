/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <cuda.h>
#include <assert.h>

#include "defines.h"
#include "gpu_utility.h"
#include "gpu_neighborList.h"

// fallback for 5.0
#if (CUDA_VERSION < 5050)
  cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority) {
    printf("WARNING: priority streams are not supported in CUDA 5.0, falling back to regular streams");
    return cudaStreamCreate(stream);
  }
#endif

void SetupGpu(int deviceId)
{
  cudaSetDevice(deviceId);
  
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  printf("Host %s using GPU %i: %s\n\n", hostname, deviceId, props.name);
}

// input is haloExchange structure for forces
// this function sets the following static GPU arrays:
//   gpu.cell_type - 0 if interior, 1 if boundary (assuming 2-rings: corresponding to boundary/interior)
//   n_boundary_cells - number of 2-ring boundary cells
//   n_boundary1_cells - number of immediate boundary cells (1 ring)
//   boundary_cells - list of boundary cells ids (2 rings)
//   interior_cells - list of interior cells ids (w/o 2 rings)
//   boundary1_cells - list of immediate boundary cells ids (1 ring)
// also it creates necessary streams
void SetBoundaryCells(SimFlat *flat, HaloExchange *hh)
{
  int nLocalBoxes = flat->boxes->nLocalBoxes;
  flat->boundary1_cells_h = (int*)malloc(nLocalBoxes * sizeof(int)); 
  int *h_boundary_cells = (int*)malloc(nLocalBoxes * sizeof(int)); 
  int *h_cell_type = (int*)malloc(nLocalBoxes * sizeof(int));
  memset(h_cell_type, 0, nLocalBoxes * sizeof(int));

  // gather data to a single list, set cell type
  int n = 0;
  ForceExchangeParms *parms = (ForceExchangeParms*)hh->parms;
  for (int ii=0; ii<6; ++ii) {
          int *cellList = parms->sendCells[ii];               
          for (int j = 0; j < parms->nCells[ii]; j++) 
                  if (cellList[j] < nLocalBoxes && h_cell_type[cellList[j]] == 0) {
                          flat->boundary1_cells_h[n] = cellList[j];
                          h_boundary_cells[n] = cellList[j];
                          h_cell_type[cellList[j]] = 1;
                          n++;
                  }
  }

  flat->n_boundary1_cells = n;
  int n_boundary1_cells = n;

  // find 2nd ring
  int neighbor_cells[N_MAX_NEIGHBORS];
  for (int i = 0; i < nLocalBoxes; i++)
    if (h_cell_type[i] == 0) {
      getNeighborBoxes(flat->boxes, i, neighbor_cells);
      for (int j = 0; j < N_MAX_NEIGHBORS; j++)
        if (h_cell_type[neighbor_cells[j]] == 1) {  
          // found connection to the boundary node - add to the list
          h_boundary_cells[n] = i;
          h_cell_type[i] = 2;
          n++;
          break;
        }
    }

  flat->n_boundary_cells = n;
  int n_boundary_cells = n;

  int n_interior_cells = flat->boxes->nLocalBoxes - n;

  // find interior cells
  int *h_interior_cells = (int*)malloc(n_interior_cells * sizeof(int));
  n = 0;
  for (int i = 0; i < nLocalBoxes; i++) {
    if (h_cell_type[i] == 0) {
      h_interior_cells[n] = i;
      n++;
    }
    else if (h_cell_type[i] == 2) {
      h_cell_type[i] = 1;
    }
  }

  // allocate on GPU
  cudaMalloc((void**)&flat->boundary1_cells_d, n_boundary1_cells * sizeof(int));
  cudaMalloc((void**)&flat->boundary_cells, n_boundary_cells * sizeof(int));
  cudaMalloc((void**)&flat->interior_cells, n_interior_cells * sizeof(int));

  // copy to GPU  
  cudaMemcpy(flat->boundary1_cells_d, flat->boundary1_cells_h, n_boundary1_cells * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(flat->boundary_cells, h_boundary_cells, n_boundary_cells * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(flat->interior_cells, h_interior_cells, n_interior_cells * sizeof(int), cudaMemcpyHostToDevice);

  // set cell types
  cudaMalloc((void**)&flat->gpu.cell_type, nLocalBoxes * sizeof(int));
  cudaMemcpy(flat->gpu.cell_type, h_cell_type, nLocalBoxes * sizeof(int), cudaMemcpyHostToDevice);

  if (flat->gpuAsync) {
    // create priority & normal streams
    cudaStreamCreateWithPriority(&flat->boundary_stream, 0, -1);	// set higher priority
    cudaStreamCreate(&flat->interior_stream);
  }
  else {
    // set streams to NULL
    flat->interior_stream = NULL;
    flat->boundary_stream = NULL;
  }

  free(h_boundary_cells);
  free(h_cell_type);
}

void AllocateGpu(SimFlat *sim, int do_eam, real_t skinDistance)
{
  int deviceId;
  struct cudaDeviceProp props;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  SimGpu *gpu = &sim->gpu;

  int total_boxes = sim->boxes->nTotalBoxes;
  int nLocalBoxes = sim->boxes->nLocalBoxes;
  int num_species = 1;

  // allocate positions, momentum, forces & energies
  int r_size = total_boxes * MAXATOMS * sizeof(real_t);
  int f_size = nLocalBoxes * MAXATOMS * sizeof(real_t);

  cudaMalloc((void**)&gpu->atoms.r.x, r_size);
  cudaMalloc((void**)&gpu->atoms.r.y, r_size);
  cudaMalloc((void**)&gpu->atoms.r.z, r_size);  

  cudaMalloc((void**)&gpu->atoms.p.x, r_size);
  cudaMalloc((void**)&gpu->atoms.p.y, r_size);
  cudaMalloc((void**)&gpu->atoms.p.z, r_size);

  cudaMalloc((void**)&gpu->atoms.f.x, f_size);
  cudaMalloc((void**)&gpu->atoms.f.y, f_size);
  cudaMalloc((void**)&gpu->atoms.f.z, f_size);

  cudaMalloc((void**)&gpu->atoms.e, f_size);
  cudaMalloc((void**)&gpu->d_updateLinkCellsRequired, sizeof(int));
  cudaMemset(gpu->d_updateLinkCellsRequired, 0, sizeof(int));

  cudaMalloc((void**)&gpu->atoms.gid, total_boxes * MAXATOMS * sizeof(int));

  // species data
  cudaMalloc((void**)&gpu->atoms.iSpecies, total_boxes * MAXATOMS * sizeof(int));
  cudaMalloc((void**)&gpu->species_mass, num_species * sizeof(real_t));

  // allocate indices, neighbors, etc.
  cudaMalloc((void**)&gpu->neighbor_cells, nLocalBoxes * N_MAX_NEIGHBORS * sizeof(int));
  cudaMalloc((void**)&gpu->neighbor_atoms, nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS * sizeof(int));
  cudaMalloc((void**)&gpu->num_neigh_atoms, nLocalBoxes * sizeof(int));

  initNeighborListGpu(&(gpu->atoms.neighborList),nLocalBoxes, skinDistance);
  initLinkCellsGpu(sim, &(gpu->boxes));

  int nMaxHaloParticles = (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes)*MAXATOMS;
  initHashTableGpu(&(gpu->d_hashTable), 2*nMaxHaloParticles);

  // total # of atoms in local boxes
  int n = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++)
    n += sim->boxes->nAtoms[iBox];
  gpu->a_list.n = n;
  cudaMalloc((void**)&gpu->a_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->a_list.cells, n * sizeof(int));

  // allocate other lists as well
  cudaMalloc((void**)&gpu->i_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->i_list.cells, n * sizeof(int));
  cudaMalloc((void**)&gpu->b_list.atoms, n * sizeof(int));
  cudaMalloc((void**)&gpu->b_list.cells, n * sizeof(int));

  // init EAM arrays
  if (do_eam)  
  {
    EamPotential* pot = (EamPotential*) sim->pot;

    cudaMalloc((void**)&gpu->eam_pot.f.values, (pot->f->n+3) * sizeof(real_t));
    cudaMalloc((void**)&gpu->eam_pot.rho.values, (pot->rho->n+3) * sizeof(real_t));
    cudaMalloc((void**)&gpu->eam_pot.phi.values, (pot->phi->n+3) * sizeof(real_t));

    cudaMalloc((void**)&gpu->eam_pot.dfEmbed, r_size);
    cudaMalloc((void**)&gpu->eam_pot.rhobar, r_size);
  }

  // initialize host data as well
  SimGpu *host = &sim->host;
  
  host->atoms.r.x=NULL; host->atoms.r.y=NULL; host->atoms.r.z=NULL;
  host->atoms.f.x=NULL; host->atoms.f.y=NULL; host->atoms.f.z=NULL;
  host->atoms.p.x=NULL; host->atoms.p.y=NULL; host->atoms.p.z=NULL;
  host->atoms.e=NULL;

  host->neighbor_cells = (int*)malloc(nLocalBoxes * N_MAX_NEIGHBORS * sizeof(int));
  host->neighbor_atoms = (int*)malloc(nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS * sizeof(int));
  host->num_neigh_atoms = (int*)malloc(nLocalBoxes * sizeof(int));

  // on host allocate list of all local atoms only
  host->a_list.atoms = (int*)malloc(n * sizeof(int));
  host->a_list.cells = (int*)malloc(n * sizeof(int));

  // temp arrays
  cudaMalloc((void**)&sim->flags, sim->boxes->nTotalBoxes * MAXATOMS * sizeof(int));
  cudaMalloc((void**)&sim->tmp_sort, sim->boxes->nTotalBoxes * MAXATOMS * sizeof(int));
  cudaMalloc((void**)&sim->gpu_atoms_buf, sim->boxes->nTotalBoxes * MAXATOMS * sizeof(AtomMsg));
  cudaMalloc((void**)&sim->gpu_force_buf, sim->boxes->nTotalBoxes * MAXATOMS * sizeof(ForceMsg));
}

void DestroyGpu(SimFlat *flat)
{
  SimGpu *gpu = &flat->gpu;
  SimGpu *host = &flat->host;

  cudaFree(gpu->d_updateLinkCellsRequired);
  cudaFree(gpu->atoms.r.x);
  cudaFree(gpu->atoms.r.y);
  cudaFree(gpu->atoms.r.z);

  cudaFree(gpu->atoms.p.x);
  cudaFree(gpu->atoms.p.y);
  cudaFree(gpu->atoms.p.z);

  cudaFree(gpu->atoms.f.x);
  cudaFree(gpu->atoms.f.y);
  cudaFree(gpu->atoms.f.z);

  cudaFree(gpu->atoms.e);

  cudaFree(gpu->atoms.gid);

  cudaFree(gpu->atoms.iSpecies);
  cudaFree(gpu->species_mass);

  cudaFree(gpu->neighbor_cells);
  cudaFree(gpu->neighbor_atoms);
  cudaFree(gpu->num_neigh_atoms);
  cudaFree(gpu->boxes.nAtoms);

  cudaFree(gpu->a_list.atoms);
  cudaFree(gpu->a_list.cells);

  cudaFree(gpu->i_list.atoms);
  cudaFree(gpu->i_list.cells);

  cudaFree(gpu->b_list.atoms);
  cudaFree(gpu->b_list.cells);

  cudaFree(flat->flags);
  cudaFree(flat->tmp_sort);
  cudaFree(flat->gpu_atoms_buf);
  cudaFree(flat->gpu_force_buf);

  if (gpu->eam_pot.f.values) cudaFree(gpu->eam_pot.f.values);
  if (gpu->eam_pot.rho.values) cudaFree(gpu->eam_pot.rho.values);
  if (gpu->eam_pot.phi.values) cudaFree(gpu->eam_pot.phi.values);

  if (gpu->eam_pot.dfEmbed) cudaFree(gpu->eam_pot.dfEmbed);
  if (gpu->eam_pot.rhobar) cudaFree(gpu->eam_pot.rhobar);


  free(host->species_mass);

  free(host->neighbor_cells);
  free(host->neighbor_atoms);
  free(host->num_neigh_atoms);

  free(host->a_list.atoms);
  free(host->a_list.cells);
}

void CopyDataToGpu(SimFlat *sim, int do_eam)
{
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // set potential
  if (do_eam) 
  {
    EamPotential* pot = (EamPotential*) sim->pot;
    gpu->eam_pot.cutoff = pot->cutoff;

    gpu->eam_pot.f.n = pot->f->n;
    gpu->eam_pot.rho.n = pot->rho->n;
    gpu->eam_pot.phi.n = pot->phi->n;

    gpu->eam_pot.f.x0 = pot->f->x0;
    gpu->eam_pot.rho.x0 = pot->rho->x0;
    gpu->eam_pot.phi.x0 = pot->phi->x0;

    gpu->eam_pot.f.xn = pot->f->x0 + pot->f->n / pot->f->invDx;
    gpu->eam_pot.rho.xn = pot->rho->x0 + pot->rho->n / pot->rho->invDx;
    gpu->eam_pot.phi.xn = pot->phi->x0 + pot->phi->n / pot->phi->invDx;

    gpu->eam_pot.f.invDx = pot->f->invDx;
    gpu->eam_pot.rho.invDx = pot->rho->invDx;
    gpu->eam_pot.phi.invDx = pot->phi->invDx;

    cudaMemcpy(gpu->eam_pot.f.values, pot->f->values-1, (pot->f->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->eam_pot.rho.values, pot->rho->values-1, (pot->rho->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->eam_pot.phi.values, pot->phi->values-1, (pot->phi->n+3) * sizeof(real_t), cudaMemcpyHostToDevice);
  }
  else
  {
    LjPotential* pot = (LjPotential*)sim->pot;
    gpu->lj_pot.sigma = pot->sigma;
    gpu->lj_pot.cutoff = pot->cutoff;
    gpu->lj_pot.epsilon = pot->epsilon;
  }

  int total_boxes = sim->boxes->nTotalBoxes;
  int nLocalBoxes = sim->boxes->nLocalBoxes;
  int r_size = total_boxes * MAXATOMS * sizeof(real_t);
  int f_size = nLocalBoxes * MAXATOMS * sizeof(real_t);
  int num_species = 1;


  for (int iBox=0; iBox < nLocalBoxes; iBox++) {
    getNeighborBoxes(sim->boxes, iBox, host->neighbor_cells + iBox * N_MAX_NEIGHBORS);

    // find itself and put first
    for (int j = 0; j < N_MAX_NEIGHBORS; j++)
      if (host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] == iBox) {
        int q = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
	host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0];
        host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0] = q;
        break;
      }
  }

  // prepare neighbor list
  for (int iBox=0; iBox < nLocalBoxes; iBox++) {
    int num_neigh_atoms = 0;
    for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
      int jBox = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
      for (int k = 0; k < sim->boxes->nAtoms[jBox]; k++) {
        host->neighbor_atoms[iBox * N_MAX_NEIGHBORS * MAXATOMS + num_neigh_atoms] = jBox * MAXATOMS + k;
        num_neigh_atoms++;
      }
    }
    host->num_neigh_atoms[iBox] = num_neigh_atoms;
  }

  // compute total # of atoms in local boxes
  int n_total = 0;
  gpu->max_atoms_cell = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    n_total += sim->boxes->nAtoms[iBox];
    if (sim->boxes->nAtoms[iBox] > gpu->max_atoms_cell)
      gpu->max_atoms_cell = sim->boxes->nAtoms[iBox];
  }
  gpu->a_list.n = n_total;
  gpu->boxes.nLocalBoxes = sim->boxes->nLocalBoxes;

  // compute and copy compact list of all atoms/cells
  int cur = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    int nIBox = sim->boxes->nAtoms[iBox];
    if (nIBox == 0) continue;
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {
      host->a_list.atoms[cur] = ii;
      host->a_list.cells[cur] = iBox;
      cur++;
    }
  }

  // initialize species
  host->species_mass = (real_t*)malloc(num_species * sizeof(real_t));
  for (int i = 0; i < num_species; i++)
    host->species_mass[i] = sim->species[i].mass;

  // copy all data to gpus
  cudaMemcpy(gpu->atoms.r.x, sim->atoms->r.x, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.r.y, sim->atoms->r.y, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.r.z, sim->atoms->r.z, r_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->atoms.p.x, sim->atoms->p.x, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.p.y, sim->atoms->p.y, r_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.p.z, sim->atoms->p.z, r_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->atoms.iSpecies, sim->atoms->iSpecies, nLocalBoxes * MAXATOMS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.gid, sim->atoms->gid, total_boxes * MAXATOMS * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->species_mass, host->species_mass, num_species * sizeof(real_t), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->neighbor_cells, host->neighbor_cells, nLocalBoxes * N_MAX_NEIGHBORS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->neighbor_atoms, host->neighbor_atoms, nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->num_neigh_atoms, host->num_neigh_atoms, nLocalBoxes * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->boxes.nAtoms, sim->boxes->nAtoms, total_boxes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->boxes.boxIDLookUp, sim->boxes->boxIDLookUp, nLocalBoxes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->boxes.boxIDLookUpReverse, sim->boxes->boxIDLookUpReverse, nLocalBoxes * sizeof(int3_t), cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->a_list.atoms, host->a_list.atoms, n_total * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->a_list.cells, host->a_list.cells, n_total * sizeof(int), cudaMemcpyHostToDevice);

}

void updateNAtomsGpu(SimFlat* sim)
{
  cudaMemcpy(sim->gpu.boxes.nAtoms,sim->boxes->nAtoms,  sim->boxes->nTotalBoxes * sizeof(int), cudaMemcpyHostToDevice);
}

void updateNAtomsCpu(SimFlat* sim)
{
  cudaMemcpy(sim->boxes->nAtoms, sim->gpu.boxes.nAtoms, sim->boxes->nTotalBoxes * sizeof(int), cudaMemcpyDeviceToHost);
}

void emptyHaloCellsGpu(SimFlat* sim)
{
  cudaMemset(sim->gpu.boxes.nAtoms + sim->boxes->nLocalBoxes, 0, (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes) * sizeof(int));
}

void GetDataFromGpu(SimFlat *sim)
{
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // copy back forces & energies
  int f_size = sim->boxes->nLocalBoxes * MAXATOMS * sizeof(real_t);

  // update num atoms
  cudaMemcpy(sim->boxes->nAtoms, gpu->boxes.nAtoms, sim->boxes->nTotalBoxes * sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(sim->atoms->p.x, gpu->atoms.p.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->p.y, gpu->atoms.p.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->p.z, gpu->atoms.p.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(sim->atoms->r.x, gpu->atoms.r.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->r.y, gpu->atoms.r.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->r.z, gpu->atoms.r.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(sim->atoms->f.x, gpu->atoms.f.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->f.y, gpu->atoms.f.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->f.z, gpu->atoms.f.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(sim->atoms->U, gpu->atoms.e, f_size, cudaMemcpyDeviceToHost);
 
  // assign energy and forces
  // compute total energy
  sim->ePotential = 0.0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    int nIBox = sim->boxes->nAtoms[iBox];
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {

      sim->ePotential += sim->atoms->U[iOff];
    }
  }
}

/// Copies positions and momentum of local particles to CPU
void GetLocalAtomsFromGpu(SimFlat *sim) 
{
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // copy back forces & energies
  int f_size = sim->boxes->nLocalBoxes * MAXATOMS * sizeof(real_t);

  cudaMemcpy(sim->atoms->p.x, gpu->atoms.p.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->p.y, gpu->atoms.p.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->p.z, gpu->atoms.p.z, f_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(sim->atoms->r.x, gpu->atoms.r.x, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->r.y, gpu->atoms.r.y, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->r.z, gpu->atoms.r.z, f_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sim->atoms->gid, gpu->atoms.gid, sim->boxes->nLocalBoxes* MAXATOMS * sizeof(int), cudaMemcpyDeviceToHost); //only req. if nlforced TODO
}

/// Compacts all atoms within the halo cells into h_compactAtoms (stored in SoA data-layout).
/// @param [in] sim
/// @param [out] h_compactAtoms stores the compacted atoms in SoA format
/// @param [out] h_cellOffset Array of at least (nHaloBoxes+1) elements will store the scan of nHaloAtoms (e.g. nAtoms(haloCell_0)=2, nAtoms(haloCell_1)=3 => h_cellOffset(0)=0,h_cellOffset(1)=3,h_cellOffset(2)=5)
int compactHaloCells(SimFlat* sim, char* h_compactAtoms, int* h_cellOffset)
{
      int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;

      
      h_cellOffset[sim->boxes->nLocalBoxes] = 0;
      for(int i = 1, iBox = sim->boxes->nLocalBoxes; i <= nHaloCells; ++i, ++iBox)
      {
         h_cellOffset[i] = sim->boxes->nAtoms[iBox] + h_cellOffset[i-1];
      }
      int nTotalAtomsInHaloCells = h_cellOffset[nHaloCells];

      AtomMsgSoA msg_h;
      getAtomMsgSoAPtr(h_compactAtoms, &msg_h, nTotalAtomsInHaloCells);

      //compact atoms from atoms struct to msg_h
      for (int ii = 0; ii < nHaloCells; ++ii)
      {
              int iOff = (sim->boxes->nLocalBoxes + ii) * MAXATOMS;
              for(int i = h_cellOffset[ii]; i < h_cellOffset[ii+1]; ++i, ++iOff)
              {
                 msg_h.rx[i] = sim->atoms->r.x[iOff];
                 msg_h.ry[i] = sim->atoms->r.y[iOff];
                 msg_h.rz[i] = sim->atoms->r.z[iOff];

                 msg_h.px[i] = sim->atoms->p.x[iOff];
                 msg_h.py[i] = sim->atoms->p.y[iOff];
                 msg_h.pz[i] = sim->atoms->p.z[iOff];

                 msg_h.type[i] = sim->atoms->iSpecies[iOff];
                 msg_h.gid[i] = sim->atoms->gid[iOff];
              }
      }
      return nTotalAtomsInHaloCells;
}

void updateGpuHalo(SimFlat *sim)
{
  //Optimization: implement version using compactHaloCells()
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;
//  char* h_compactAtoms = sim->atomExchange->recvBufM;
//  int*  h_cellOffset = ((AtomExchangeParms*)sim->atomExchange->parms)->h_natoms_buf;
//
//  int nTotalAtomsInHaloCells= compactHaloCells( sim, h_compactAtoms, h_cellOffset); 
//    
//  //copy compacted atoms to gpu
//  char* d_compactAtoms = sim->gpu_atoms_buf;
//  cudaMemcpy((void*)(d_compactAtoms), h_compactAtoms, nTotalAtomsInHaloCells * sizeof(AtomMsg), cudaMemcpyHostToDevice);
//
//  //alias host and device buffers with AtomMsgSoA
//  AtomMsgSoA msg_d;
//  getAtomMsgSoAPtr(d_compactAtoms, &msg_d, nTotalAtomsInHaloCells);
//
//  //copy cellOffset to cpu
//  int* d_cellOffset = ((AtomExchangeParms*)sim->atomExchange->parms)->d_natoms_buf;
//  cudaMemcpy(d_cellOffsets, h_cellOffset, (nHaloCells+1) * sizeof(int), cudaMemcpyHostToDevice);
//
//  const int blockDim = 256;
//  int grid = nTotalAtomsInHaloCells + (blockDim - 1) / blockDim;
//  unpackHaloCells<<<grid, block>>>(d_cellOffset, d_compactAtoms, sim->gpu); //TODO implement this function

  int f_size = nHaloCells * MAXATOMS * sizeof(real_t);
  int i_size = nHaloCells * MAXATOMS * sizeof(int);

  cudaMemcpy(gpu->atoms.p.x+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->p.x+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.p.y+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->p.y+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.p.z+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->p.z+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->atoms.r.x+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->r.x+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.r.y+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->r.y+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu->atoms.r.z+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->r.z+(sim->boxes->nLocalBoxes * MAXATOMS), f_size, cudaMemcpyHostToDevice);

  cudaMemcpy(gpu->atoms.gid+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->gid+(sim->boxes->nLocalBoxes * MAXATOMS), i_size, cudaMemcpyHostToDevice); //TODO REmove?
  cudaMemcpy(gpu->atoms.iSpecies+(sim->boxes->nLocalBoxes * MAXATOMS), sim->atoms->iSpecies+(sim->boxes->nLocalBoxes * MAXATOMS), i_size, cudaMemcpyHostToDevice); //TODO remove?
}

void initLinkCellsGpu(SimFlat *sim, LinkCellGpu* boxes)
{

  boxes->nTotalBoxes = sim->boxes->nTotalBoxes;
  boxes->nLocalBoxes = sim->boxes->nLocalBoxes;

  boxes->gridSize.x = sim->boxes->gridSize[0];
  boxes->gridSize.y = sim->boxes->gridSize[1];
  boxes->gridSize.z = sim->boxes->gridSize[2];

  boxes->localMin.x = sim->boxes->localMin[0];
  boxes->localMin.y = sim->boxes->localMin[1];
  boxes->localMin.z = sim->boxes->localMin[2];

  boxes->localMax.x = sim->boxes->localMax[0];
  boxes->localMax.y = sim->boxes->localMax[1];
  boxes->localMax.z = sim->boxes->localMax[2];

  boxes->invBoxSize.x = sim->boxes->invBoxSize[0];
  boxes->invBoxSize.y = sim->boxes->invBoxSize[1];
  boxes->invBoxSize.z = sim->boxes->invBoxSize[2];

  assert (sim->boxes->nLocalBoxes == sim->boxes->gridSize[0] * sim->boxes->gridSize[1] * sim->boxes->gridSize[2]);
  cudaMalloc((void**)&boxes->nAtoms, sim->boxes->nTotalBoxes * sizeof(int));
  cudaMalloc((void**)&boxes->boxIDLookUpReverse, sim->boxes->nLocalBoxes * sizeof(int3_t));
  cudaMalloc((void**)&boxes->boxIDLookUp, sim->boxes->nLocalBoxes * sizeof(int));
}
