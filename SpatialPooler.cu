#include <cstring>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "curand_kernel.h"

/*
 * This should be an per block agent synchronizing with others only on arrival of new input
 * Global parameters (density, inc, dec, ...) are passed as parameters, not saved to global memory (faster)
 * The whole input is copied to global memory, each block loads a part of the input
 * Each thread is responsible for one column (gets rid of all annoying for loops over cols)
 * Multidim input cannot be coalesced into a single vector output, since we want to preserve topology
 * over allocated blocks
 * Memory: input, overlaps, permanences, potential conn., connected counts, active cols,
 * boost factors, active/overlap duty cycles + their mins is changed each cycle, but can't fit - have to
 * use sparse matrices/vectors (thrust, cuSPARSE)
 * Luckily, there needs to be no communication between threads
 * We have to think aboout implementing this as a backend to the original class - can be satisfied by
 * passing all relevant members via an arg struct
 */

using namespace std;

typedef unsigned int UInt;
typedef double Real;

struct args
{
	// Parameters
    bool* gInput;
    UInt inputPitch;
    UInt stimulusThreshold;
    Real potentialPct;
    Real connectedPct;
    Real synPermTrimThreshold;
    Real synPermMax;
    Real synPermConnected;

	// Constants
	UInt MAX_CONNECTED;
	UInt IN_BLOCK_SIZE; // Dims of input chunk for each cuda block

	// Global memory pointers
    bool* in_dev;
    bool* cols_dev;
	UInt* pot_dev;
	Real* per_dev;
	Real* overlap_dc_dev; // odc serve to maintain same act. freq. for each col. (per block)
	Real* active_dc_dev; // adc serve to compute boost factors

	// Array pitches
	size_t pot_pitch_in_bytes;
	size_t per_pitch_in_bytes;
	size_t odc_pitch_in_bytes;
	size_t adc_pitch_in_bytes;
};


__device__
void calculateOverlap(bool* input, UInt* pot_dev, Real* per_dev, UInt* olaps_sh, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
    olaps_sh[tx] = 0;
    for(int i=0; i < MAX_CONNECTED; i++)
    {
		int in_idx = pot_dev[tx*MAX_CONNECTED+i];
		if(input[inBlockSize*blockIdx.x + in_idx] && per_dev[tx*MAX_CONNECTED+i] >= threshold)
        	olaps_sh[tx]++;
    }
}

__device__
void inhibitColumns(UInt* olaps_sh, bool* cols_dev, bool &active)
{
    int tx = threadIdx.x;
	int numLarger = 0;
	active = false;
	
	for(int i=0; i < blockDim.x; i++)
	{
		if(olaps_sh[i] > olaps_sh[tx]) numLarger++;
	}
	if(numLarger < 0.02*blockDim.x) active = true;

	__syncthreads();

	cols_dev[blockIdx.x*blockDim.x + tx] = active;
}

__global__
void compute(args ar)
{
    // x - column, y - row in cuda
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int inX = blockIdx.x*blockDim.x+tx;
    int inY = blockIdx.y*blockDim.y+ty;

	bool active = false;

	// Pointers to dynamically shared memory are all given the same address
    extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];

	// Assign a thread in each block to read data from potential matrix, input and permanences into local memory ?

	calculateOverlap(ar.in_dev, ar.pot_dev, ar.per_dev, olaps_sh, ar.synPermConnected, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	__syncthreads();

	// Choose winning columns (with most overlaps):
	// Each thread goes through the overlaps stored in shared array, computes the number of cols with larger overlaps and if this is < sparsity*blockSize -> active
	inhibitColumns(olaps_sh, ar.cols_dev, active);
	
	__syncthreads();

	// printf(" %d ", olaps_sh[tx]);


	// Active cols will increment permanences of their active connections / decrement inactive

	// Update duty cycles - what info do we need? duty cycle period, iteration number, overlap and activity for each col in this and previous step.
	// __syncthreads() to make sure all cols computed its duty cycles
	// Duty cycles are average activities / overlaps of each column. To compute boost factors, we need average activity of the whole block.
	// odc: periodically check, if lower than a min, all cols get permanence incremented 
	// adc: to compute boost factors
	// So either one or all threads will go through the shared memory array and compute the average, then each thread computes the boost factor

	// All necessary info is written to global memory, shared memory is cleared.

}
