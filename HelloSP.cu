#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <assert.h>

#include "SpatialPooler.cu"

using namespace std;

typedef unsigned int UInt;
typedef float Real;

UInt* generatePotentialPools(int cols, const UInt IN_BLOCK_SIZE, Real potentialPct, const UInt MAX_CONNECTED, UInt* numPotential)
{
    UInt* potentialPools = new UInt[cols*MAX_CONNECTED];
    int connected = 0;
    for(int i=0; i < cols; i++)
    {
    	connected = 0;
		// Generated indeces should be in (0,IN_BLOCK_SIZE) and their count should be <= MAX_CONNECTED and around potentialPct*IN_BLOCK_SIZE
        for(int j=0; j < IN_BLOCK_SIZE; j++)
        {
            if((Real)(rand()%100)/100 <= potentialPct && connected < MAX_CONNECTED)
            {
                potentialPools[i*MAX_CONNECTED + connected++] = j; 
            }
        }
		numPotential[i] = connected;
    }
    return potentialPools;
}

Real initPermanencesConnected(Real synPermConnected_, Real synPermMax_)
{
	Real p = synPermConnected_ +
	             (synPermMax_ - synPermConnected_)*((Real)((rand()%100))/100);
	return p;
}

Real initPermanencesNotConnected(Real synPermConnected_)
{
	Real p = synPermConnected_ * (Real)((rand()%100))/100;
	return p;
}

Real* generatePermanences(int cols, int inputSize, UInt* potential, Real connectedPct,
		Real synPermConnected_, Real synPermMax_, const UInt MAX_CONNECTED, UInt* numPotential,
	   	const UInt BLOCK_SIZE, const UInt IN_BLOCK_SIZE)
{
    Real* permanences = new Real[cols*MAX_CONNECTED];
	int connected = 0;
	int curr_block = 0;
    bool found = false;

	for(int i=0; i < cols; i++)
	{
		connected = 0;
		// We need to only go through the input block corresponding to the current column
		// This means we need to convert current column to the input block number
		curr_block = floor(i / BLOCK_SIZE);
		// j is the global index of connection in the input matrix
		for(int j=curr_block*IN_BLOCK_SIZE; j < curr_block*IN_BLOCK_SIZE + IN_BLOCK_SIZE; j++)
		{
			// Find if this input is potentially connected with this column
			found=false;
            for(int k=0; k < numPotential[i]; k++)
            {
                if(potential[i*MAX_CONNECTED+k] == j % IN_BLOCK_SIZE) {
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
                    permanences[i*MAX_CONNECTED+connected++] = initPermanencesConnected(synPermConnected_, synPermMax_);
                }
                else
                {
                    permanences[i*MAX_CONNECTED+connected++] = initPermanencesNotConnected(synPermConnected_);
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
				connected_arr[i][connected++] = j;
			}
		}
	}
	return connected_arr;
}

void generate01(bool* ar, size_t size, Real inDensity)
{
	for(int i=0; i < size; i++)
	{
		ar[i] = (Real)(rand()%100)/100 <= inDensity ? 1 : 0;
	}
}

void visualize_input(bool* in_host, UInt* potentialPools, Real* permanences, UInt* numPotential, const UInt IN_SIZE, const UInt SP_SIZE, const UInt IN_BLOCK_SIZE, const UInt MAX_CONNECTED)
{
	printf("POTENTIAL CONNECTIONS WITH PERMANENCES\n");
	for(int i=0; i<SP_SIZE; i++)
	{
		for(int j=0; j<MAX_CONNECTED; j++)
			printf("%d \t", potentialPools[i*MAX_CONNECTED+j]);
		printf("\n");
		for(int j=0; j<numPotential[i]; j++)
			printf("%.2f\t", permanences[i*MAX_CONNECTED+j]);
		printf("\n");
		printf("%d \n", numPotential[i]);
	}

	printf("INPUT SDR\n");
	for(int i=0; i<IN_SIZE; i++)
	{
		printf("%d ", in_host[i]);
		if(i % IN_BLOCK_SIZE == 0 && i > 0)
			printf("\n");
	}
	printf("\n");
}

void visualize_output(bool* cols_host, const UInt SP_SIZE)
{
	// The final sparsity will approach target with increasing block size
	int ones = 0;
	for(int i=0; i < SP_SIZE; i++)
		if(cols_host[i] > 0) ones++;
	printf("Sparsity: %f \n", (Real)ones/SP_SIZE);
}

void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ERROR  : %s\n", cudaGetErrorString(error));
    printf("==================================================\n");
}

int main(int argc, const char * argv[])
{
	const UInt SP_SIZE = 524288;
	const UInt IN_SIZE = 1048576;
	const UInt BLOCK_SIZE = 64; // Two warps
	const UInt NUM_BLOCKS = SP_SIZE/BLOCK_SIZE;
	const UInt IN_BLOCK_SIZE = IN_SIZE/NUM_BLOCKS; // Size of chunk of input processed by a single cuda block
	const UInt MAX_CONNECTED = 16;
    const Real IN_DENSITY = 0.5; // Density of input connections
    srand(time(NULL));

	size_t sm = BLOCK_SIZE*(sizeof(Real) + sizeof(UInt));

    // construct input args
    args ar;
	ar.iteration_num=0;
	ar.learn=true;
	ar.localAreaDensity=0.02; // SP density after inhibition
    ar.potentialPct=0.5; // 
    ar.connectedPct=0.5;
    ar.stimulusThreshold=0;
    ar.synPermTrimThreshold=0.025;
    ar.synPermMax=1.0;
    ar.synPermConnected=0.1;
	ar.synPermActiveInc=0.05;
	ar.synPermInactiveDec=0.008;
	ar.synPermBelowStimulusInc=ar.synPermConnected / 10.0;
	ar.dutyCyclePeriod=1000;
	ar.boostStrength=0.05; // 0 means no boosting
	ar.minOdc=0; // maxOcd * minPctOdc
	ar.minPctOdc=0.001;
	ar.update_period=50;
	ar.SP_SIZE = SP_SIZE;
	ar.MAX_CONNECTED = MAX_CONNECTED;
	ar.IN_BLOCK_SIZE = IN_BLOCK_SIZE;

	// Host memory pointers
    bool* cols_host = new bool[SP_SIZE];
	bool* in_host = new bool[IN_SIZE];
    UInt* potentialPools;
	Real* permanences;
	Real* boosts = new Real[SP_SIZE*MAX_CONNECTED];
	UInt* numPotential = new UInt[SP_SIZE];
	UInt* numConnected = new UInt[SP_SIZE];

	// Host memory allocation	
	std::fill_n(boosts, SP_SIZE*MAX_CONNECTED, 1);
	std::fill_n(numPotential, SP_SIZE, 0);
	std::fill_n(numConnected, SP_SIZE, 0);

	potentialPools = generatePotentialPools(SP_SIZE, IN_BLOCK_SIZE, ar.potentialPct, MAX_CONNECTED, numPotential);
	permanences = generatePermanences(SP_SIZE, IN_SIZE, potentialPools, ar.connectedPct, ar.synPermConnected, ar.synPermMax, MAX_CONNECTED, numPotential,
					BLOCK_SIZE, IN_BLOCK_SIZE);
	generate01(in_host, IN_SIZE, IN_DENSITY);

	// visualize_input(in_host, potentialPools, permanences, numPotential, IN_SIZE, SP_SIZE, IN_BLOCK_SIZE, MAX_CONNECTED);

	// Global memory pointers
	args* ar_dev;

	// Global memory allocation
	cudaError_t result;
    result = cudaMalloc((void **) &ar_dev, sizeof(ar)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.in_dev, IN_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.olaps_dev, SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.cols_dev, SP_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
	result = cudaMalloc((void **) &ar.numPot_dev, SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.pot_dev, MAX_CONNECTED*SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0); // width, height, x, y 
    result = cudaMalloc((void **) &ar.per_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.odc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.adc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.boosts_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 

	// Memcpy to device
    result = cudaMemcpy(ar_dev, &ar, sizeof(ar), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.in_dev, in_host, IN_SIZE*sizeof(bool), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.numPot_dev, numPotential, SP_SIZE*sizeof(UInt), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.pot_dev, potentialPools, MAX_CONNECTED*SP_SIZE*sizeof(UInt), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.per_dev, permanences, MAX_CONNECTED*SP_SIZE*sizeof(Real), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.boosts_dev, boosts, MAX_CONNECTED*SP_SIZE*sizeof(Real), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);

	// Kernel call
    compute<<<NUM_BLOCKS, BLOCK_SIZE, sm>>>(ar_dev);

    // Memcpy from device
    result = cudaMemcpy(cols_host, ar.cols_dev, SP_SIZE*sizeof(bool), cudaMemcpyDeviceToHost); if(result) printErrorMessage(result, 0); 

	visualize_output(cols_host, SP_SIZE);

    cudaFree(ar.in_dev); cudaFree(ar.cols_dev); cudaFree(ar.pot_dev); cudaFree(ar.per_dev); cudaFree(ar.boosts_dev);
	cudaFree(ar.odc_dev); cudaFree(ar.adc_dev); cudaFree(ar.numPot_dev);

    return 0;
}
