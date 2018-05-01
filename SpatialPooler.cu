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
void inhibitColumns(UInt* olaps_sh, bool* cols_dev, Real* active_sh, bool &active, Real sparsity)
{
    int tx = threadIdx.x;
	int numLarger = 0;
	active = false;
	
	for(int i=0; i < blockDim.x; i++)
	{
		if(olaps_sh[i] > olaps_sh[tx]) numLarger++;
	}
	if(numLarger < sparsity*blockDim.x) active = true;

	__syncthreads();

	cols_dev[blockIdx.x*blockDim.x + tx] = active;
	active_sh[tx] = active;
}

// TODO: Can this be implemented via matrix (element-wise) multiplication?
__device__
void adaptSynapses(bool* input, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool active, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
   	int inX = blockDim.x*blockIdx.x + tx;
	if(active)
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

__device__
void updateDutyCycles(Real* odc_dev, Real* adc_dev, UInt* olaps_sh, bool active, UInt iteration_num, UInt dutyCyclePeriod)
{
    int tx = threadIdx.x;

	// Let grow divisor only to a dutyCyclePeriod to not make the update increasingly negligible
	Real period = dutyCyclePeriod > iteration_num ? iteration_num : dutyCyclePeriod;

	odc_dev[blockDim.x*blockIdx.x+tx] = (odc_dev[blockDim.x*blockIdx.x+tx]*(period-1) + (Real)(olaps_sh[tx] > 0)) / period;
	adc_dev[blockDim.x*blockIdx.x+tx] = (odc_dev[blockDim.x*blockIdx.x+tx]*(period-1) + (Real)active) / period;
}

__device__
void averageActivity(Real* active_sh)
{
	Real avg = 0;
	for(int i=0; i < blockDim.x; i++)
	{
		avg += active_sh[i];
	}
	active_sh[threadIdx.x] = avg / (Real)blockDim.x;
}

__device__
void averageActivityReduction(Real* active_sh)
{
	int tx = threadIdx.x;
	UInt BLOCK_SIZE = blockDim.x;
	
	if(BLOCK_SIZE >= 512)
	{ 
		if(tx < 256) 
		{ 
			active_sh[tx] += active_sh[tx+256]; 
		} 
		__syncthreads(); 
	}
    if(BLOCK_SIZE >= 256)
   	{ 
		if(tx < 128) 
		{ 
			active_sh[tx] += active_sh[tx+128]; 
		} 
		__syncthreads(); 
	}
    if(BLOCK_SIZE >= 128)
   	{ 
		if(tx < 64) 
		{ 
			active_sh[tx] += active_sh[tx+64]; 
		} 
		__syncthreads(); 
	}

    if(tx < 32) 
    {
        if(BLOCK_SIZE >= 64) 
			active_sh[tx] += active_sh[tx+32];
        if(BLOCK_SIZE >= 32) 
			active_sh[tx] += active_sh[tx+16];
        if(BLOCK_SIZE >= 16) 
			active_sh[tx] += active_sh[tx+8];
        if(BLOCK_SIZE >= 8) 
			active_sh[tx] += active_sh[tx+4];
        if(BLOCK_SIZE >= 4)
			active_sh[tx] += active_sh[tx+2];
        if(BLOCK_SIZE >= 2) 
			active_sh[tx] += active_sh[tx+1];
    }

	__syncthreads();

	// According to https://devblogs.nvidia.com/using-shared-memory-cuda-cc/, this should result in a broadcast
    active_sh[tx] = active_sh[0] / BLOCK_SIZE;
}

__device__
void updateBoosts(Real* adc_dev, Real* boosts_dev, Real targetDensity, Real boostStrength)
{
    int inX = blockIdx.x*blockDim.x+threadIdx.x;
	boosts_dev[inX] = exp((targetDensity - adc_dev[inX])*boostStrength);
}

__device__
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
__device__
void updateMinOdc(Real* odc_dev, Real &minOdc, Real minPctOdc, const UInt SP_SIZE)
{
	Real maxOdc = 0;
	for(int i=0; i<SP_SIZE; i++)
		maxOdc = odc_dev[i] > maxOdc ? odc_dev[i] : maxOdc;
	minOdc = minPctOdc * maxOdc;
}

__global__
void compute(args* ar_ptr)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) 
		ar_ptr->iteration_num++;
	
	args ar = *ar_ptr;

	bool active = false;
	Real avg_act = 0;

    extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];
	Real* active_sh = (Real*)&shared[blockDim.x];

	calculateOverlap(ar.in_dev, ar.pot_dev, ar.per_dev, ar.boosts_dev, ar.numPot_dev, olaps_sh, ar.synPermConnected, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	__syncthreads();

	inhibitColumns(olaps_sh, ar.cols_dev, active_sh, active, ar.localAreaDensity);
	
	__syncthreads();

	adaptSynapses(ar.in_dev, ar.pot_dev, ar.per_dev, ar.synPermActiveInc, ar.synPermInactiveDec, active, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	updateDutyCycles(ar.odc_dev, ar.adc_dev, olaps_sh, active, ar.iteration_num, ar.dutyCyclePeriod);

	// active_sh will hold average activity per block for each column
	averageActivityReduction(active_sh);

	__syncthreads();

	updateBoosts(ar.adc_dev, ar.boosts_dev, avg_act, ar.boostStrength);

	bumpUpColumnsWithWeakOdc(ar.odc_dev, ar.per_dev, ar.numPot_dev, ar.minOdc, ar.synPermBelowStimulusInc, ar.MAX_CONNECTED);

	if(ar.iteration_num % ar.update_period == 0)
		updateMinOdc(ar.odc_dev, ar.minOdc, ar.minPctOdc, ar.SP_SIZE);
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
	Real* active_sh = (Real*) &olaps_sh[BLOCK_SIZE];

	olaps_sh[threadIdx.x] = olaps_dev[threadIdx.x];

	bool active = false;

	__syncthreads();

	inhibitColumns(olaps_sh, cols_dev, active_sh, active, localAreaDensity);
}

__global__
void adaptSynapses_wrapper(bool* in_dev, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool* active_arr, const UInt IN_BLOCK_SIZE, const UInt MAX_CONNECTED, const UInt SP_SIZE)
{
	int inX = blockIdx.x*blockDim.x + threadIdx.x;
	if(inX < SP_SIZE)
	{
		bool active = active_arr[inX];
		adaptSynapses(in_dev, pot_dev, per_dev, synPermActiveInc, synPermInactiveDec, active, IN_BLOCK_SIZE, MAX_CONNECTED);
	}
}

__global__
void averageActivity_wrapper(bool* cols_dev, Real* avg_dev)
{
	int tx = threadIdx.x;

	extern __shared__ UInt shared[];
	Real* active_sh = (Real*) &shared[0];

	active_sh[tx] = (Real) cols_dev[tx];

	averageActivityReduction(active_sh);

	avg_dev[tx] = active_sh[tx];	
}
