#include <stdio.h>

using namespace std;

typedef unsigned int UInt;
typedef float Real;

struct args
{
	// Parameters
    UInt inputPitch;
    UInt stimulusThreshold;
    Real potentialPct;
    Real connectedPct;
	Real localAreaDensity;
    Real synPermTrimThreshold;
    Real synPermMax;
    Real synPermConnected;
	Real synPermActiveInc;
	Real synPermInactiveDec;
	Real synPermBelowStimulusInc;
	UInt dutyCyclePeriod;
	Real boostStrength;
	Real minOdc;
	Real minPctOdc;
	bool learn;

	// Global memory pointers
	bool* in_dev;
    bool* cols_dev;
	UInt* olaps_dev;
	UInt* pot_dev;
	Real* per_dev;
	Real* boosts_dev;
	Real* odc_dev; // odc serve to maintain same act. freq. for each col. (per block)
	Real* adc_dev; // adc serve to compute boost factors
	UInt* numPot_dev;
	Real* avg_act_dev;

	// Constants
	UInt SP_SIZE;
	UInt MAX_CONNECTED;
	UInt IN_BLOCK_SIZE; // Dims of input chunk for each cuda block

	// Array pitches
	size_t pot_pitch_in_bytes;
	size_t per_pitch_in_bytes;
	size_t odc_pitch_in_bytes;
	size_t adc_pitch_in_bytes;
	size_t bst_pitch_in_bytes;

	// Bookkeeping vars
	UInt iteration_num;
	UInt update_period;
};


// TODO: This could be done via parallel and distributed matrix multiplication.
__device__
void calculateOverlap(bool* input, UInt* pot_dev, Real* per_dev, Real* boosts_dev, UInt* numPot_dev, UInt* olaps_sh, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED)
{ 
	int tx = threadIdx.x;
   	int inX = blockDim.x*blockIdx.x + tx; // Global index in the SP
    olaps_sh[tx] = 0;
	// TODO: This reading from global memory is inefficient. Maybe need more efficient data structures to fit into shared memory?
    for(int i=0; i < numPot_dev[inX]; i++)
    {
		UInt in_idx = pot_dev[inX*MAX_CONNECTED+i]; // Index of block-specific input
		if(input[inBlockSize*blockIdx.x + in_idx])
				if(per_dev[inX*MAX_CONNECTED+i] >= threshold)
        			olaps_sh[tx] += boosts_dev[inX+i];
    }
}

// TODO: This could be done via parallel sorting.
__device__
void inhibitColumns(UInt* olaps_sh, bool* cols_dev, Real sparsity)
{
    int tx = threadIdx.x;
	int numLarger = 0;
	bool active = false;
	
	for(int i=0; i < blockDim.x; i++)
	{
		if(olaps_sh[i] > olaps_sh[tx]) numLarger++;
	}
	if(numLarger < sparsity*blockDim.x) active = true;

	__syncthreads();

	cols_dev[blockIdx.x*blockDim.x + tx] = active;
}

// TODO: Can this be implemented via matrix (element-wise) multiplication?
__global__
void adaptSynapses(bool* input, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool* cols_dev, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
   	int inX = blockDim.x*blockIdx.x + tx;
	if(cols_dev[inX])
	{
		for(int i=0; i < MAX_CONNECTED; i++)
    	{
			int in_idx = pot_dev[inX*MAX_CONNECTED+i];
			if(input[inBlockSize*blockIdx.x + in_idx])
				per_dev[inX*MAX_CONNECTED+i] = max(min(1.0, per_dev[inX*MAX_CONNECTED+i]+synPermActiveInc), 0.0);
			else
				per_dev[inX*MAX_CONNECTED+i] = max(min(1.0, per_dev[inX*MAX_CONNECTED+i]-synPermInactiveDec), 0.0);
    	}
	}
}

__global__
void updateDutyCycles(Real* odc_dev, Real* adc_dev, UInt* olaps_dev, bool* cols_dev, UInt iteration_num, UInt dutyCyclePeriod)
{
    int tx = threadIdx.x;
   	int inX = blockDim.x*blockIdx.x + tx;
	bool active = cols_dev[inX];

	// Let grow divisor only to a dutyCyclePeriod to not make the update increasingly negligible
	Real period = dutyCyclePeriod > iteration_num ? iteration_num : dutyCyclePeriod;

	odc_dev[blockDim.x*blockIdx.x+tx] = (odc_dev[blockDim.x*blockIdx.x+tx]*(period-1) + (Real)(olaps_dev[inX] > 0)) / period;
	adc_dev[blockDim.x*blockIdx.x+tx] = (odc_dev[blockDim.x*blockIdx.x+tx]*(period-1) + (Real)active) / period;
}

// TODO: This can be done via reduction.
__global__
void averageActivity(bool* cols_dev, Real* avg_act_dev)
{
	Real avg = 0;
	int start = blockDim.x*blockIdx.x;
	int inX = blockDim.x*blockIdx.x + threadIdx.x;
	for(int i=0; i < blockDim.x; i++)
	{
		avg += (Real)cols_dev[start+i];
	}
	avg /= (Real)blockDim.x;
	avg_act_dev[inX] = avg;
}

__global__
void updateBoosts(Real* adc_dev, Real* boosts_dev, Real* targetDensity, Real boostStrength)
{
    int inX = blockIdx.x*blockDim.x+threadIdx.x;
	boosts_dev[inX] = exp((targetDensity[inX] - adc_dev[inX])*boostStrength);
}

__global__
void bumpUpColumnsWithWeakOdc(Real* odc_dev, Real* per_dev, UInt* numPot, Real minOdc, Real synPermBelowStimulusInc, const UInt MAX_CONNECTED)
{
	int tx = threadIdx.x;
    int inX = blockIdx.x*blockDim.x+tx;

	if(odc_dev[inX] < minOdc) {
		for(int i=0; i<numPot[inX]; i++)
			per_dev[tx*MAX_CONNECTED+i] += synPermBelowStimulusInc;
	}
}

// TODO: This can be done via reduction.
__global__
void updateMinOdc(Real* odc_dev, Real &minOdc, Real minPctOdc, const UInt SP_SIZE)
{
	Real maxOdc = 0;
	for(int i=0; i<SP_SIZE; i++)
		maxOdc = odc_dev[i] > maxOdc ? odc_dev[i] : maxOdc;
	minOdc = minPctOdc * maxOdc;
}

__global__
void overlap_inhibit(args* ar_ptr)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) 
		ar_ptr->iteration_num++;
	
	args ar = *ar_ptr;

    extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];


	calculateOverlap(ar.in_dev, ar.pot_dev, ar.per_dev, ar.boosts_dev, ar.numPot_dev, olaps_sh, ar.synPermConnected, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	__syncthreads();

	inhibitColumns(olaps_sh, ar.cols_dev, ar.localAreaDensity);
	
	// if(ar.iteration_num % ar.update_period == 0)
	//	updateMinOdc(ar.odc_dev, ar.minOdc, ar.minPctOdc, ar.SP_SIZE);
}




__global__
void calculateOverlap_wrapper(bool* input, UInt* pot_dev, Real* per_dev, Real* boosts_dev, UInt* numPot_dev, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED, UInt* olaps_dev, const UInt SP_SIZE)
{
	extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];

	calculateOverlap(input, pot_dev, per_dev, boosts_dev, numPot_dev, olaps_sh, threshold, inBlockSize, MAX_CONNECTED);

	if(blockDim.x*blockIdx.x+threadIdx.x < SP_SIZE)
		olaps_dev[blockDim.x*blockIdx.x+threadIdx.x] = olaps_sh[threadIdx.x];
}


__global__
void inhibitColumns_wrapper(UInt* olaps_dev, bool* cols_dev, Real localAreaDensity, const UInt BLOCK_SIZE)
{
	extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];

	olaps_sh[threadIdx.x] = olaps_dev[threadIdx.x];

	__syncthreads();

	inhibitColumns(olaps_sh, cols_dev, localAreaDensity);
}

// __global__
// void adaptSynapses_wrapper(bool* in_dev, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool* active_arr, const UInt IN_BLOCK_SIZE, const UInt MAX_CONNECTED, const UInt SP_SIZE)
// {
// 	int inX = blockIdx.x*blockDim.x + threadIdx.x;
// 	if(inX < SP_SIZE)
// 	{
// 		adaptSynapses(in_dev, pot_dev, per_dev, synPermActiveInc, synPermInactiveDec, active_arr, IN_BLOCK_SIZE, MAX_CONNECTED);
// 	}
// }
