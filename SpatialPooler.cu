#include <stdio.h>

using namespace std;

typedef unsigned int UInt;
typedef float Real;

// Define global constants
const UInt SP_SIZE = 131072;
const UInt IN_SIZE = 262144;
const UInt BLOCK_SIZE = 64; // Two warps
const UInt NUM_BLOCKS = SP_SIZE/BLOCK_SIZE;
const UInt IN_BLOCK_SIZE = IN_SIZE/NUM_BLOCKS; // Size of chunk of input processed by a single cuda block
const UInt MAX_CONNECTED = 16;
const Real IN_DENSITY = 0.5; // Density of input connections

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
	Real minPctOdc;
	bool learn;

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

__device__
void calculateOverlap(bool* in_dev, bool* in_sh, UInt* pot_dev, Real* per_dev, Real* boosts_dev, UInt* numPot_dev, UInt* olaps_sh, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED)
{ 
	int tx = threadIdx.x;
   	int sp_idx = blockDim.x*blockIdx.x + tx; // Global index in the SP
	int num_assnd = (int)(inBlockSize/blockDim.x);
	int in_idx = inBlockSize*blockIdx.x + tx*num_assnd; // Beginning of portion of input assigned to this thread
    olaps_sh[tx] = 0;
	// Let each thread load its part of input to shared memory
	for(int i=0; i < num_assnd; i++)
		in_sh[tx*num_assnd+i] = in_dev[in_idx+i]; 

	__syncthreads();

    for(int i=0; i < numPot_dev[sp_idx]; i++)
    {
		UInt bl_idx = pot_dev[sp_idx*MAX_CONNECTED+i]; // Index of block-specific input
		if(in_sh[bl_idx] && per_dev[sp_idx*MAX_CONNECTED+i] >= threshold)
        	olaps_sh[tx] += boosts_dev[sp_idx+i];
    }
}

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

__device__
void adaptSynapses(bool* in_dev, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool active, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
   	int sp_idx = blockDim.x*blockIdx.x + tx;
	if(active)
	{
		for(int i=0; i < MAX_CONNECTED; i++)
    	{
			int in_idx = pot_dev[sp_idx*MAX_CONNECTED+i];
			if(in_dev[inBlockSize*blockIdx.x + in_idx])
				per_dev[sp_idx*MAX_CONNECTED+i] = min(1.0, per_dev[sp_idx*MAX_CONNECTED+i]+synPermActiveInc);
			else
				per_dev[sp_idx*MAX_CONNECTED+i] = max(per_dev[sp_idx*MAX_CONNECTED+i]-synPermInactiveDec, 0.0);
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
    int sp_idx = blockIdx.x*blockDim.x+threadIdx.x;
	boosts_dev[sp_idx] = exp((targetDensity - adc_dev[sp_idx])*boostStrength);
}

__device__
void bumpUpColumnsWithWeakOdc(Real* odc_dev, Real* per_dev, UInt* numPot, Real* minOdc_dev, Real synPermBelowStimulusInc, const UInt MAX_CONNECTED)
{
	int tx = threadIdx.x;
    int sp_idx = blockIdx.x*blockDim.x+tx;

	if(odc_dev[sp_idx] < minOdc_dev[blockIdx.x]) {
		for(int i=0; i<numPot[sp_idx]; i++)
			per_dev[tx*MAX_CONNECTED+i] += synPermBelowStimulusInc;
	}
}

__device__
void updateMinOdc(Real* odc_dev, Real* odc_sh, Real* minOdc_dev, Real minPctOdc, const UInt SP_SIZE)
{
	Real maxOdc = 0;
	for(int i=0; i<SP_SIZE; i++)
		maxOdc = odc_dev[i] > maxOdc ? odc_dev[i] : maxOdc;
	if(threadIdx.x == 0)
		minOdc_dev[blockIdx.x] = minPctOdc * maxOdc;
}

// Lambdas are available in nvcc > 7.5
__device__
void updateMinOdcReduction(Real* odc_dev, Real* odc_sh, Real* minOdc_dev, Real minPctOdc, const UInt SP_SIZE)
{
	int tx = threadIdx.x;
	int sp_idx = blockDim.x*blockIdx.x + threadIdx.x;
	UInt BLOCK_SIZE = blockDim.x;

	odc_sh[tx] = odc_dev[sp_idx];
	
	if(BLOCK_SIZE >= 512)
	{ 
		if(tx < 256) 
		{ 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+256]); 
		} 
		__syncthreads(); 
	}
    if(BLOCK_SIZE >= 256)
   	{ 
		if(tx < 128) 
		{ 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+128]); 
		} 
		__syncthreads(); 
	}
    if(BLOCK_SIZE >= 128)
   	{ 
		if(tx < 64) 
		{ 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+64]); 
		} 
		__syncthreads(); 
	}

    if(tx < 32) 
    {
        if(BLOCK_SIZE >= 64) 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+32]);
        if(BLOCK_SIZE >= 32) 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+16]);
        if(BLOCK_SIZE >= 16) 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+8]);
        if(BLOCK_SIZE >= 8) 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+4]);
        if(BLOCK_SIZE >= 4)
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+2]);
        if(BLOCK_SIZE >= 2) 
			odc_sh[tx] = max(odc_sh[tx], odc_sh[tx+1]);
    }

	if(threadIdx.x == 0)
		minOdc_dev[blockIdx.x] = minPctOdc * odc_sh[0];
}


__device__
void generatePotentialPools(UInt* pot_dev, UInt* pot_dev_nums, Real potentialPct)
{
	
}

__device__
void generatePermanences()
{
}

__global__
void compute(args* ar_ptr, void* data)
{
	// Global memory pointers
    bool* cols_dev = (bool*) data;
	bool* in_dev = &cols_dev[SP_SIZE];
	UInt* pot_dev = (UInt*) &in_dev[IN_SIZE];
	UInt* numPot_dev = &pot_dev[SP_SIZE*MAX_CONNECTED];
	Real* per_dev = (Real*) &numPot_dev[SP_SIZE];
	Real* boosts_dev = &per_dev[SP_SIZE*MAX_CONNECTED];
	UInt* olaps_dev = (UInt*) &boosts_dev[SP_SIZE*MAX_CONNECTED];
	Real* odc_dev = (Real*) &olaps_dev[SP_SIZE]; // odc serve to maintain same act. freq. for each col. (per block)
	Real* adc_dev =  &odc_dev[MAX_CONNECTED*SP_SIZE]; // adc serve to compute boost factors
	Real* minOdc_dev = &adc_dev[MAX_CONNECTED*SP_SIZE]; // Stores minumum overlap duty cycles per block 

	
	if (blockIdx.x == 0 && threadIdx.x == 0) 
		ar_ptr->iteration_num++;
	
	args ar = *ar_ptr;

	bool active = false;
	Real avg_act = 0;

    extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];
	// Lepsi jako volatile
	Real* active_sh = (Real*)&shared[blockDim.x];
	Real* odc_sh = &active_sh[blockDim.x];
	bool* in_sh = (bool*) &odc_sh[blockDim.x];

	calculateOverlap(in_dev, in_sh, pot_dev, per_dev, boosts_dev, numPot_dev, olaps_sh, ar.synPermConnected, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	__syncthreads();

	inhibitColumns(olaps_sh, cols_dev, active_sh, active, ar.localAreaDensity);
	
	__syncthreads();

	adaptSynapses(in_dev, pot_dev, per_dev, ar.synPermActiveInc, ar.synPermInactiveDec, active, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	updateDutyCycles(odc_dev, adc_dev, olaps_sh, active, ar.iteration_num, ar.dutyCyclePeriod);

	averageActivityReduction(active_sh);

	__syncthreads();

	updateBoosts(adc_dev, boosts_dev, avg_act, ar.boostStrength);

	bumpUpColumnsWithWeakOdc(odc_dev, per_dev, numPot_dev, minOdc_dev, ar.synPermBelowStimulusInc, ar.MAX_CONNECTED);

	if(ar.iteration_num % ar.update_period == 0)
		updateMinOdc(odc_dev, odc_sh, minOdc_dev, ar.minPctOdc, ar.SP_SIZE);
}

__global__
void calculateOverlap_wrapper(bool* in_dev, UInt* pot_dev, Real* per_dev, Real* boosts_dev, UInt* numPot_dev, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED, UInt* olaps_dev, const UInt SP_SIZE)
{
	extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];
	bool* in_sh = (bool*) &olaps_sh[blockDim.x];

	calculateOverlap(in_dev, in_sh, pot_dev, per_dev, boosts_dev, numPot_dev, olaps_sh, threshold, inBlockSize, MAX_CONNECTED);

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
	int sp_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(sp_idx < SP_SIZE)
	{
		bool active = active_arr[sp_idx];
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
