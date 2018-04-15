#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <assert.h>

// #include "SpatialPooler.cu"

using namespace std;

typedef unsigned int UInt;
typedef double Real;

// This is problematic - we need to keep track of how many input bits have we connected :/
// But it should pay off if MAX_CONNECTED << inputSize
// Simpler alternative is a binary matrix numCols x inBlockSize (a col is connected locally to in block size)
UInt** generatePotentialPools(int cols, const UInt IN_BLOCK_SIZE, Real potentialPct, const UInt MAX_CONNECTED, UInt* numPotential)
{
    UInt** potentialPools = new UInt*[cols];
    int connected = 0;
    for(int i=0; i < cols; i++)
    {
    	connected = 0;
        potentialPools[i] = new UInt[MAX_CONNECTED];
		// Generated indeces should be in (0,IN_BLOCK_SIZE) and their count should be <= MAX_CONNECTED and around potentialPct*IN_BLOCK_SIZE
        for(int j=0; j < IN_BLOCK_SIZE; j++)
        {
            if((Real)(rand()%100)/100 <= potentialPct && connected <= MAX_CONNECTED)
            {
                potentialPools[i][connected++] = j;
                numPotential[i]++;
            }
        }
    }
    return potentialPools;
}

Real initPermanencesConnected(Real synPermConnected_, Real synPermMax_)
{
	Real p = synPermConnected_ +
	             (synPermMax_ - synPermConnected_)*(Real)((rand()%100)/100);
	return p;
}

Real initPermanencesNotConnected(Real synPermConnected_)
{
	Real p = synPermConnected_ * (Real)((rand()%100)/100);
	return p;
}

Real** generatePermanences(int cols, int inputSize, UInt** potential, Real connectedPct,
		Real synPermConnected_, Real synPermMax_, const UInt MAX_CONNECTED, UInt* numPotential,
	   	const UInt BLOCK_SIZE, const UInt IN_BLOCK_SIZE)
{
    Real** permanences = new Real*[cols];
	int connected = 0;
	int curr_block = 0;
    bool found = false;

	for(int i=0; i < cols; i++)
	{
		connected = 0;
        permanences[i] = new Real[MAX_CONNECTED];
		// We need to only go through the input block corresponding to the current column
		// This means we need to convert current column to the input block number
		curr_block = i % BLOCK_SIZE;
		for(int j=curr_block*IN_BLOCK_SIZE; j < curr_block*IN_BLOCK_SIZE + IN_BLOCK_SIZE; j++)
		{
			// Find if this input is potentially connected with this column
			found=false;
            for(int k=0; k < numPotential[i]; k++)
            {
                if(potential[i][k] == j % IN_BLOCK_SIZE) {
					found = true;
					break;
				}
            }
			// If there is, decide if it will be. The structure of the data is as follows:
		    // potential[col][index of the synapse on the segment] = index of input in the block
			// permanences[col][index of the synapse on the segment] = permanence of the synapse
            if(found)
            {
                if((Real)(rand()%100)/100 <= connectedPct)
                {
                    permanences[i][connected++] = initPermanencesConnected(synPermConnected_, synPermMax_);
                }
                else
                {
                    permanences[i][connected++] = initPermanencesNotConnected(synPermConnected_);
                }
            }
		}
	}
	return permanences;
}

// TO BE DELETED
// There should also be a parameter to raise permanences so that minimum number of synapses is connected.
UInt** computeConnected(Real** permanences, UInt** potential, UInt cols, UInt inputSize,
		Real synPermConnected_, const UInt MAX_CONNECTED, UInt* numPotential)
{
	UInt** connected_arr = new UInt*[cols];
	int connected = 0;
	for(int i=0; i < inputSize; i++)
	{
		connected = 0;
        connected_arr[i] = new UInt[MAX_CONNECTED];
		for(int j=0; j < numPotential[i]; j++)
		{
			if(permanences[i][j] < synPermConnected_)
			{
				connected_arr[i][connected++] = j; // SISEGF
			}
		}
	}
	return connected_arr;
}

void generate01(bool* ar, size_t size, Real inDensity)
{
	for(int i=0; i < size; i++)
	{
		ar[i] = (rand()%100)/100 <= inDensity ? 1 : 0;
	}
}

void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
    printf("Wished allocated memory : %d\n", memorySize);
    printf("==================================================\n");
}

struct args
{
    UInt inputDim[2];
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

int main(int argc, const char * argv[])
{
    const UInt DIM_SP = 256;
    const UInt DIM_INPUT = 512;
    const UInt DIM_BLOCK = 16; // dimensions of cuda block
    const UInt IN_DIM_BLOCK = DIM_INPUT/(DIM_SP/DIM_BLOCK); // Dimension of chunk of input processed by a single cuda block
    const UInt MAX_CONNECTED = IN_DIM_BLOCK*IN_DIM_BLOCK/4; // A hard limit o # of conn. input bits / column
    const Real IN_DENSITY = 0.5;

	const UInt SP_SIZE = DIM_SP * DIM_SP;
	const UInt IN_SIZE = DIM_INPUT * DIM_INPUT;
	const UInt BLOCK_SIZE = DIM_BLOCK * DIM_BLOCK;
	const UInt IN_BLOCK_SIZE = IN_DIM_BLOCK * IN_DIM_BLOCK;
    srand(time(NULL));

    UInt colsDim[] = {DIM_SP, DIM_SP};
    // dim3 grid_dm(DIM_SP/DIM_BLOCK, DIM_SP/DIM_BLOCK, 1);
    // dim3 block_dm(DIM_BLOCK, DIM_BLOCK, 1);

    bool* cols_host = new bool[SP_SIZE];
	bool* in_host = new bool[IN_SIZE];

	// Shared mem.: permanences, potential conn., active/overlap duty cycles, active cols - copied from global
	// 				overlaps, connected counts, boost factors + their mins (using active d.c.) - computed locally
    // TODO: Compute the mem. requirements, allocate proper amount
	// input chunk, overlaps, connected, boost factors
	size_t sm = (IN_BLOCK_SIZE)*sizeof(bool) + (2*BLOCK_SIZE)*sizeof(UInt) + (BLOCK_SIZE)*sizeof(Real);

    // construct input args
    args ar;
    ar.inputDim[0] = DIM_INPUT; ar.inputDim[1] = DIM_INPUT; // Dimensions of the chunk of input each block works with
    ar.potentialPct=0.5;
    ar.connectedPct=0.5;
    ar.stimulusThreshold_=0;
    ar.synPermTrimThreshold_=0.025;
    ar.synPermMax_=1.0;
    ar.synPermConnected_=0.1;
	// Why am I passing these as arguments??? Should be rather copied to global memory
    // ar.gInput = new bool[IN_SIZE]; 
    // generate01(ar.gInput, DIM_INPUT*DIM_INPUT, IN_DENSITY);
    // ar.permanences_; // Allocated further
    // ar.potentialPools_; // Allocated further
    // ar.connectedSynapses_;

    UInt** potentialPools = new UInt*[SP_SIZE];
    for(int i = 0; i < DIM_SP*DIM_SP; ++i) {
        potentialPools[i] = new UInt[MAX_CONNECTED];
    }
    //UInt** connected = new UInt*[SP_SIZE];
    //for(int i = 0; i < DIM_SP*DIM_SP; ++i) {
    //    connected[i] = new UInt[MAX_CONNECTED];
    //}
	Real** permanences = new Real*[SP_SIZE]; // potential array provides indexes
    for(int i = 0; i < DIM_SP*DIM_SP; ++i) {
        permanences[i] = new Real[MAX_CONNECTED];
    }

	UInt numPotential[SP_SIZE] = {0};
	UInt numConnected[SP_SIZE] = {0};
	potentialPools = generatePotentialPools(SP_SIZE, IN_BLOCK_SIZE, ar.potentialPct, MAX_CONNECTED, numPotential);
	permanences = generatePermanences(SP_SIZE, IN_SIZE, potentialPools, ar.connectedPct, ar.synPermConnected_, ar.synPermMax_, MAX_CONNECTED, numPotential,
					BLOCK_SIZE, IN_BLOCK_SIZE);
	// Connected don't need to be copied to device - it's enough to allocate memory for it and compute on device
	// connected = computeConnected(permanences, potentialPools, SP_SIZE, IN_SIZE, ar.synPermConnected_, MAX_CONNECTED, numPotential);
	generate01(in_host, IN_SIZE, IN_DENSITY);


	// Let's start with 1D computation only - we have CC3.0, so this is not a limitation
	// Hyperparameters will be passed to global memory as a struct
	// TODO: Make a list of necessary parameters to compute(), make a complete arg struct
	// Or maybe first write a sketch of the compute function and then decide what needs to be passed in?
    cudaError_t result;
    bool* in_dev;
    bool* cols_dev;
	UInt** pot_dev;
	Real** per_dev;
	Real* overlap_dc_dev; // odc serve to maintain same act. freq. for each col. (per block)
	Real* active_dc_dev; // adc serve to compute boost factors

	size_t pot_pitch_in_bytes;
	size_t per_pitch_in_bytes;
	size_t odc_pitch_in_bytes;
	size_t adc_pitch_in_bytes;

    result = cudaMalloc((void **) &in_dev, IN_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &cols_dev, SP_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
    result = cudaMallocPitch((void **) &pot_dev, &pot_pitch_in_bytes, MAX_CONNECTED*sizeof(UInt), SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0); // width, height, x, y 
    result = cudaMallocPitch((void **) &per_dev, &per_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMallocPitch((void **) &overlap_dc_dev, &odc_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMallocPitch((void **) &active_dc_dev, &adc_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 

    result = cudaMemcpy((void **) &in_dev, &in_host, IN_SIZE*sizeof(bool), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    cudaMemcpy2D((void **) &pot_dev, pot_pitch_in_bytes, &potentialPools, MAX_CONNECTED*sizeof(UInt), MAX_CONNECTED*sizeof(UInt), SP_SIZE, cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    cudaMemcpy2D((void **) &per_dev, per_pitch_in_bytes, &permanences, MAX_CONNECTED*sizeof(Real), MAX_CONNECTED*sizeof(Real), SP_SIZE, cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);

    compute<<<grid_dm, block_dm, sm>>>(ar);

    // // Memcpy from device
    // result = cudaMemcpy2D(cols_host, cols_pitch_in_bytes, cols_dev,
    //     		colsDim[0] * sizeof(bool), colsDim[1] * sizeof(bool), cudaMemcpyDeviceToHost);
    
    cudaFree(in_dev); cudaFree(cols_dev); cudaFree(pot_dev); cudaFree(per_dev); 

    return 0;
}
