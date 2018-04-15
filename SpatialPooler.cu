#include <cstring>
#include <iostream>
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
using namespace nupic;

struct args
{
    vector<UInt> inputDimensions;
    bool* gInput;
    UInt inputPitch;
    UInt stimulusThreshold_;
    Real potentialPct;
    Real connectedPct;
    Real synPermTrimThreshold_;
    Real synPermMax_;
    Real synPermConnected_;

    int inBlockSizeX;
    int inBlockSizeY;

    bool* input_;
    bool* potentialPools_;
    Real* permanences_;
    bool* connectedSynapses_;
    UInt* overlaps_;
};


// Connect each column randomly to input
__device__
void init_potential(args* in)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int inputSize = in->inBlockSizeX*in->inBlockSizeY;
    curandState s;
    curand_init(452342,blockIdx.x * blockDim.x + threadIdx.x,0,&s);

    for(int i=0; i < inputSize; i++)
    {
        // Connect potentialPct % of the input randomly
        if(curand_uniform_double(&s) <= in->potentialPct)
        {
            in->potentialPools_[(blockDim.y*ty+tx)*inputSize + i] = 1;
        }
        else
        {
            in->potentialPools_[(blockDim.y*ty+tx)*inputSize + i] = 0;
        }
    }
}

__device__
Real initPermConnected_(args* in)
{
    curandState s;
    curand_init(312312,blockIdx.x * blockDim.x + threadIdx.x,0,&s);
    Real p = in->synPermConnected_ +
             (in->synPermMax_ - in->synPermConnected_)*(Real)curand_uniform_double(&s);
    return p;
}

__device__
Real initPermNonConnected_(args* in)
{
    curandState s;
    curand_init(564564,blockIdx.x * blockDim.x + threadIdx.x,0,&s);
    Real p = in->synPermConnected_ * (Real)curand_uniform_double(&s);
    return p;
}

// Init permananences randomly and assign connections according to threshold
__device__
void init_permanences(args* in)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int inputSize = in->inBlockSizeX*in->inBlockSizeY;

    curandState s;
    curand_init(979123,blockIdx.x * blockDim.x + threadIdx.x,0,&s);

    for(int i=0; i < inputSize; i++)
    {
        if(in->potentialPools_[(blockDim.y*ty+tx)*inputSize + i] < 1)
        {
            continue;
        }
        if(curand_uniform_double(&s) <= in->connectedPct)
        {
            in->permanences_[(blockDim.y*ty+tx)*inputSize + i] = initPermConnected_(in);
        }
        else
        {
            in->permanences_[(blockDim.y*ty+tx)*inputSize + i] = initPermNonConnected_(in);
        }
        in->permanences_[(blockDim.y*ty+tx)*inputSize + i] =
                in->permanences_[(blockDim.y*ty+tx)*inputSize + i] < in->synPermTrimThreshold_ ? 0 : in->permanences_[(blockDim.y*ty+tx)*inputSize + i];
    }
}

__device__
void initialize(args* in)
{
    init_potential(in);
    init_permanences(in);
    // TODO: Update active according to permanences
}

__device__
void calculateOverlap(bool* input, bool* connectedSynapses_, UInt* overlaps_, UInt inputSize)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    overlaps_[ty*blockDim.y + tx] = 0;
    for(int i=0; i < inputSize; i++)
    {
        overlaps_[ty*blockDim.x + tx] +=
                connectedSynapses_[(blockDim.y*ty+tx)*inputSize + i] * input[i];
    }
}

__global__
void compute(args* ar)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // x - column, y - row in cuda
    int inBlockSizeX = ceil(in->inputDimensions[1]/gridDim.x);
    int inBlockSizeY = ceil(in->inputDimensions[0]/gridDim.y);
    int inX = blockIdx.x*blockDim.x+tx;
    int inY = blockIdx.y*blockDim.y+ty;

    extern __shared__ bool* shared;
    bool* input_ = shared;
    bool* connectedSynapses_ = (bool*) input_[blockDim.x*blockDim.y];
    bool* potentialPools_ = &connectedSynapses_[blockDim.x*blockDim.y*inBlockSizeX*inBlockSizeY];
    Real* permanences_ = (Real*)&potentialPools_[blockDim.x*blockDim.y*inBlockSizeX*inBlockSizeY];
    UInt* overlaps_ = (UInt*) &permanences_[blockDim.x*blockDim.y];

    in->input_ = input_;
    in->connectedSynapses_ = connectedSynapses_;
    in->potentialPools_ = potentialPools_;
    in->permanences_ = permanences_;
    in->overlaps_ = overlaps_;
    in->inBlockSizeX = inBlockSizeX;
    in->inBlockSizeY = inBlockSizeY;
    initialize(in);

    if(tx < inBlockSizeX && ty < inBlockSizeY)
    {
        input_[ty*inBlockSizeX+tx] = in->gInput[inY*in->inputPitch+inX];
        calculateOverlap(input_, connectedSynapses_, overlaps_, inBlockSizeX*inBlockSizeY);
    }

	// Assign a thread in each block to read data from potential matrix, input and permanences into local memory

	// Each thread will access a row in potential matrix and go through the indeces there (lengths of the for loops will be very similar -> no slowdown)
	// Each index corresponds to an input in the local input matrix
	// Then it reads permanence corresponding to the local input in the local permanences matrix (same index as the index 
	// of the index in the potential matrix), and if bigger than a threshold -> increases the columns overlap (not by 1, but by boost factor)
	// These (boosted) overlaps need to be stored in shared memory. Now, we need to select k cols with the largest overlap.
	// __syncthreads() to make sure all threads (cols) computed its boosted overlap
	// Each thread goes through the overlaps stored in shared array, computes the number of cols with larger overlaps and if this is < sparsity*blockSize -> active

	// Active cols will increment permanences of their active connections / decrement inactive

	// Update duty cycles - what info do we need? duty cycle period, iteration number, overlap and activity for each col in this and previous step.
	// __syncthreads() to make sure all cols computed its duty cycles
	// Duty cycles are average activities / overlaps of each column. To compute boost factors, we need average activity of the whole block.
	// odc: periodically check, if lower than a min, all cols get permanence incremented 
	// adc: to compute boost factors
	// So either one or all threads will go through the shared memory array and compute the average, then each thread computes the boost factor

	// All necessary info is written to global memory, shared memory is cleared.

}
