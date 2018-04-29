#include "SpatialPooler.cu"
#include <assert.h>
#include <stdio.h>
#include <algorithm>

using namespace std;

template <typename T>
bool compare(const T* corr_vec, const T* out_vec, UInt size)
{
	for(int i=0; i < size; i++)
	{
		printf("%d \t %d \n", corr_vec[i], out_vec[i]);
		// printf("%d, ", out_vec[i]);
		if(corr_vec[i] != out_vec[i]) 
			return false;
	}
	return true;
}

void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ERROR  : %s\n", cudaGetErrorString(error));
    printf("==================================================\n");
}


void setup_device2D(args& ar, bool* in_host, UInt* numPotential, UInt* potentialPools, Real* permanences, Real* boosts, const UInt SP_SIZE, const UInt IN_SIZE, const UInt MAX_CONNECTED)
{
    cudaError_t result;
    // result = cudaMalloc((void **) &ar_dev, sizeof(ar)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.in_dev, IN_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.cols_dev, SP_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
	result = cudaMalloc((void **) &ar.numPot_dev, SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0);
    result = cudaMallocPitch((void **) &ar.pot_dev, &ar.pot_pitch_in_bytes, MAX_CONNECTED*sizeof(UInt), SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0); // width, height, x, y 
    result = cudaMallocPitch((void **) &ar.per_dev, &ar.per_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMallocPitch((void **) &ar.odc_dev, &ar.odc_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMallocPitch((void **) &ar.adc_dev, &ar.adc_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMallocPitch((void **) &ar.boosts_dev, &ar.bst_pitch_in_bytes, MAX_CONNECTED*sizeof(Real), SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 

	// Memcpy to device
    // result = cudaMemcpy(ar_dev, &ar, sizeof(ar), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.in_dev, in_host, IN_SIZE*sizeof(bool), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.numPot_dev, numPotential, SP_SIZE*sizeof(UInt), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy2D(ar.pot_dev, ar.pot_pitch_in_bytes, potentialPools, MAX_CONNECTED*sizeof(UInt), MAX_CONNECTED*sizeof(UInt), SP_SIZE, cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy2D(ar.per_dev, ar.per_pitch_in_bytes, permanences, MAX_CONNECTED*sizeof(Real), MAX_CONNECTED*sizeof(Real), SP_SIZE, cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy2D(ar.boosts_dev, ar.bst_pitch_in_bytes, boosts, MAX_CONNECTED*sizeof(Real), MAX_CONNECTED*sizeof(Real), SP_SIZE, cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
}

void setup_device1D(args& ar, bool* in_host, UInt* numPotential, UInt* potentialPools, Real* permanences, Real* boosts, const UInt SP_SIZE, const UInt IN_SIZE, const UInt MAX_CONNECTED)
{
    cudaError_t result;
    // result = cudaMalloc((void **) &ar_dev, sizeof(ar)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.in_dev, IN_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.olaps_dev, SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0);
    // result = cudaMalloc((void **) &ar.cols_dev, SP_SIZE*sizeof(bool)); if(result) printErrorMessage(result, 0);
	result = cudaMalloc((void **) &ar.numPot_dev, SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0);
    result = cudaMalloc((void **) &ar.pot_dev, MAX_CONNECTED*SP_SIZE*sizeof(UInt)); if(result) printErrorMessage(result, 0); // width, height, x, y 
    result = cudaMalloc((void **) &ar.per_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.odc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.adc_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 
    result = cudaMalloc((void **) &ar.boosts_dev, MAX_CONNECTED*SP_SIZE*sizeof(Real)); if(result) printErrorMessage(result, 0); 

	// Memcpy to device
    // result = cudaMemcpy(ar_dev, &ar, sizeof(ar), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.in_dev, in_host, IN_SIZE*sizeof(bool), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.numPot_dev, numPotential, SP_SIZE*sizeof(UInt), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.pot_dev, potentialPools, MAX_CONNECTED*SP_SIZE*sizeof(UInt), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.per_dev, permanences, MAX_CONNECTED*SP_SIZE*sizeof(Real), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
    result = cudaMemcpy(ar.boosts_dev, boosts, MAX_CONNECTED*SP_SIZE*sizeof(Real), cudaMemcpyHostToDevice); if(result) printErrorMessage(result, 0);
}



void testCalculateOverlap()
{
	const UInt SP_SIZE = 16;
	const UInt IN_SIZE = 32;
	const UInt BLOCK_SIZE = 8;
	const UInt NUM_BLOCKS = SP_SIZE/BLOCK_SIZE;
	const UInt MAX_CONNECTED = 4;
	const UInt IN_BLOCK_SIZE = IN_SIZE/NUM_BLOCKS;
	Real threshold = 0.1;

							//0, 1, 2, 3, 4, 5, 6, 7
	bool in_host[IN_SIZE] =	{ 0, 1, 0, 1, 0, 1, 0, 1,
							//8, 9, 1, 1, 2, 3, 4, 5
		   		 	   		  1, 0, 1, 0, 1, 0, 1, 0,
   					   		  1, 1, 1, 1, 0, 0, 0, 0,
					   		  0, 0, 0, 0, 1, 1, 1, 1	
							};

	UInt potential[SP_SIZE*MAX_CONNECTED] = 	{ 0, 2, 3, 5,
   						 					  1, 3, 4, 7,
						 					  2, 5, 6, 7,
						 					  1, 4, 5, 11,
						 					  3, 10, 11, 15,
						 					  1, 9, 12, 14,
						 					  0, 13, 14, 15,
   						 					  1, 8, 9, 12, // 1st block
						 					  2, 5, 6, 7,
						 					  1, 4, 5, 6,
						 					  3, 4, 6, 7,
						 					  1, 11, 13, 14,
						 					  0, 8, 10, 15,
   						 					  1, 9, 10, 11,
						 					  2, 5, 9, 12,
						 					  1, 4, 5, 13, // 2nd block
											 };

	Real permanences[SP_SIZE*MAX_CONNECTED] = 	{ 0.09, 0.11, 0.09, 0.11, 
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
											  0.09, 0.11, 0.09, 0.11,
   											};

	Real boosts[SP_SIZE*MAX_CONNECTED];
	std::fill_n(boosts, SP_SIZE*MAX_CONNECTED, 1);

	UInt numPot[SP_SIZE];
	std::fill_n(numPot, SP_SIZE, MAX_CONNECTED);

	UInt correct_overlaps[SP_SIZE] = { 1, 2, 2, 0, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0, 1, 1 }; 
	// std::fill_n(correct_overlaps, SP_SIZE, 0);

	UInt olaps[SP_SIZE];
	
	args ar;

	setup_device1D(ar, in_host, numPot, potential, permanences, boosts, SP_SIZE, IN_SIZE, MAX_CONNECTED);

	calculateOverlap_wrapper<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(UInt)>>>(ar.in_dev, ar.pot_dev, ar.per_dev, ar.boosts_dev, ar.numPot_dev, threshold, IN_BLOCK_SIZE, MAX_CONNECTED, ar.olaps_dev, SP_SIZE);

	cudaError_t result = cudaMemcpy(olaps, ar.olaps_dev, SP_SIZE*sizeof(UInt), cudaMemcpyDeviceToHost); if(result) printErrorMessage(result, 0);

	assert(compare<UInt>(correct_overlaps, olaps, SP_SIZE));
	// compare<UInt>(correct_overlaps, olaps, SP_SIZE);
}

int main(int argc, const char * argv[])
{
	testCalculateOverlap();
}
