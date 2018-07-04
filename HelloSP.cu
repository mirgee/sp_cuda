#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>
#include <assert.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>

#include "SpatialPooler.cu"

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

typedef unsigned int UInt;
typedef float Real;

UInt* generatePotentialPools(UInt* potentialPools, int cols, const UInt IN_BLOCK_SIZE, Real potentialPct, const UInt MAX_CONNECTED, UInt* numPotential)
{
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

Real* generatePermanences(Real* permanences, int cols, int inputSize, UInt* potential, Real connectedPct,
		Real synPermConnected_, Real synPermMax_, const UInt MAX_CONNECTED, UInt* numPotential,
	   	const UInt BLOCK_SIZE, const UInt IN_BLOCK_SIZE)
{
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

bool* generate01(bool* ar, size_t size, Real inDensity)
{
	for(int i=0; i < size; i++)
	{
		ar[i] = (Real)(rand()%100)/100 <= inDensity ? 1 : 0;
	}
	return ar;
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
	srand(time(NULL));
	
	curandState dev_states;
	

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
	ar.minPctOdc=0.001;
	ar.update_period=50;
	ar.SP_SIZE = SP_SIZE;
	ar.MAX_CONNECTED = MAX_CONNECTED;
	ar.IN_BLOCK_SIZE = IN_BLOCK_SIZE;

	ar.num_connected = std::floor(MAX_CONNECTED*ar.connectedPct);

	// Host memory allocation
	size_t host_alloc_size = (IN_SIZE+SP_SIZE)*sizeof(bool);
    bool* cols_host = (bool*) malloc(host_alloc_size);
	bool* in_host = (bool*) &cols_host[SP_SIZE]; 

	// Host memory init	
	in_host = generate01(in_host, IN_SIZE, IN_DENSITY);

	// visualize_input(in_host, potentialPools, permanences, numPotential, IN_SIZE, SP_SIZE, IN_BLOCK_SIZE, MAX_CONNECTED);

	// Global memory pointers
	args* ar_dev;

	// Global memory allocation
    checkError( cudaMalloc((void **) &ar_dev, sizeof(ar)) );

	checkError( cudaMallocPitch((void **) &ar.pot_dev, &ar.pot_dev_pitch, ar.num_connected, SP_SIZE) );
	checkError( cudaMallocPitch((void **) &ar.per_dev, &ar.per_dev_pitch, ar.num_connected, SP_SIZE) );
	checkError( cudaMalloc((void **) &ar.boosts_dev, SP_SIZE*ar.num_connected*sizeof(Real)) );
    checkError( cudaMalloc((void **) &ar.in_dev, IN_SIZE*sizeof(bool)) ); 
    checkError( cudaMalloc((void **) &ar.olaps_dev, SP_SIZE*sizeof(UInt)) );
    checkError( cudaMalloc((void **) &ar.cols_dev, SP_SIZE*sizeof(bool)) );
	checkError( cudaMalloc((void **) &ar.numPot_dev, SP_SIZE*sizeof(UInt)) );
    checkError( cudaMalloc((void **) &ar.odc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)) );
    checkError( cudaMalloc((void **) &ar.adc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)) );
	checkError( cudaMalloc((void **) &ar.minOdc_dev, NUM_BLOCKS*sizeof(Real)) );

	// Global memory initialization
	// Potential pools
	thrust::device_vector<UInt> input_indeces(IN_BLOCK_SIZE);
	thrust::sequence(input_indeces.begin(), input_indeces.end(), 0, 1);

	size_t sm = BLOCK_SIZE*sizeof(UInt);
	generatePotentialPools<<<SP_SIZE, BLOCK_SIZE, sm>>>(ar.pot_dev, ar.pot_dev_pitch, ar.num_connected, thrust::raw_pointer_cast(input_indeces.data()), &dev_states);

	// Permanences
	generatePermanences<<<SP_SIZE, ar.num_connected>>>(ar.per_dev, ar.per_dev_pitch, ar.connectedPct, ar.synPermConnected, ar.synPermMax, &dev_states);

	// Boosts
	thrust::device_ptr<float> dev_ptr(ar.boosts_dev);
	thrust::fill(dev_ptr, dev_ptr+SP_SIZE*ar.num_connected*sizeof(Real), 1.0);

	// Memcpy to device
    checkError( cudaMemcpy(ar_dev, (void**) &ar, sizeof(ar), cudaMemcpyHostToDevice) );
    checkError( cudaMemcpy(ar.in_dev, in_host, IN_SIZE*sizeof(bool), cudaMemcpyHostToDevice) );

	// Compute permanences
	sm = IN_BLOCK_SIZE*sizeof(UInt);
	calculateOverlap<<<SP_SIZE, BLOCK_SIZE, sm>>>(ar.in_dev, ar.pot_dev, ar.pot_dev_pitch, ar.per_dev, ar.per_dev_pitch, ar.boosts_dev, ar.synPermConnected, ar.num_connected)
	
	// Kernel call
	// sm = BLOCK_SIZE*(2*sizeof(Real) + sizeof(UInt)) + IN_BLOCK_SIZE*sizeof(bool);
    // compute<<<NUM_BLOCKS, BLOCK_SIZE, sm>>>(ar_dev, data_dev);

    // // Memcpy from device
    // checkError( cudaMemcpy(cols_host, data_dev, SP_SIZE*sizeof(bool), cudaMemcpyDeviceToHost)); 

	// visualize_output(cols_host, SP_SIZE);

    // cudaFree(ar_dev); cudaFree(data_dev);

    return 0;
}
