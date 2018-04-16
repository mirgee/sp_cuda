#include <stdio.h>

using namespace std;

typedef unsigned int UInt;
typedef double Real;

struct args
{
	// Parameters
    UInt inputPitch;
    UInt stimulusThreshold;
    Real potentialPct;
    Real connectedPct;
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

	// Constants
	UInt SP_SIZE;
	UInt MAX_CONNECTED;
	UInt IN_BLOCK_SIZE; // Dims of input chunk for each cuda block

	// Global memory pointers
    bool* in_dev;
    bool* cols_dev;
	UInt* pot_dev;
	Real* per_dev;
	Real* boosts_dev;
	Real* odc_dev; // odc serve to maintain same act. freq. for each col. (per block)
	Real* adc_dev; // adc serve to compute boost factors
	UInt* numPot_dev;

	// Array pitches
	size_t pot_pitch_in_bytes;
	size_t per_pitch_in_bytes;
	size_t odc_pitch_in_bytes;
	size_t adc_pitch_in_bytes;

	// Bookkeeping vars
	UInt iteration_num;
	UInt update_period;
};


__device__
void calculateOverlap(bool* input, UInt* pot_dev, Real* per_dev, Real* boosts_dev, UInt* numPot_dev, UInt* olaps_sh, Real threshold, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
	int inX = blockDim.x*blockIdx.x + tx;
	Real boostFactor = boosts_dev[inX];
    olaps_sh[tx] = 0;
    for(int i=0; i < numPot_dev[inX]; i++)
    {
		int in_idx = pot_dev[tx*MAX_CONNECTED+i];
		if(input[inBlockSize*blockIdx.x + in_idx] && per_dev[tx*MAX_CONNECTED+i] >= threshold)
        	olaps_sh[tx] += boostFactor;
    }
}

// TODO: This could be done via parallel sorting.
__device__
void inhibitColumns(UInt* olaps_sh, bool* cols_dev, bool* active_sh, bool &active)
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
	active_sh[tx] = active;
}

// TODO: Can this be implemented via matrix (element-wise) multiplication?
__device__
void adaptSynapses(bool* input, UInt* pot_dev, Real* per_dev, Real synPermActiveInc, Real synPermInactiveDec, bool active, const UInt inBlockSize, const UInt MAX_CONNECTED)
{
    int tx = threadIdx.x;
	if(active)
	{
		for(int i=0; i < MAX_CONNECTED; i++)
    	{
			int in_idx = pot_dev[tx*MAX_CONNECTED+i];
			if(input[inBlockSize*blockIdx.x + in_idx])
				per_dev[tx*MAX_CONNECTED+i] += synPermActiveInc;
			else
				per_dev[tx*MAX_CONNECTED+i] -= synPermInactiveDec;
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

// TODO: This can be done via reduction.
__device__
void averageActivity(bool* active_sh, Real &avg)
{
	avg = 0;
	for(int i=0; i < blockDim.x; i++)
	{
		avg += (Real)active_sh[i];
	}
	avg /= (Real)blockDim.x;
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
void compute(args ar)
{
	ar.iteration_num++;
	bool active = false;
	Real avg_act = 0;

    extern __shared__ UInt shared[];
	UInt* olaps_sh = &shared[0];
	bool* active_sh = (bool*)&shared[blockDim.x];

	// TODO: Decide on what to copy to local memory. What would be the task size limit implications?
	// TODO: When do we need to synchronize threads?

	calculateOverlap(ar.in_dev, ar.pot_dev, ar.per_dev, ar.boosts_dev, ar.numPot_dev, olaps_sh, ar.synPermConnected, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);

	__syncthreads();

	inhibitColumns(olaps_sh, ar.cols_dev, active_sh, active);
	
	__syncthreads();

	// printf(" %d ", olaps_sh[tx]);

	adaptSynapses(ar.in_dev, ar.pot_dev, ar.per_dev, ar.synPermActiveInc, ar.synPermInactiveDec, active, ar.IN_BLOCK_SIZE, ar.MAX_CONNECTED);
	
	__syncthreads();

	updateDutyCycles(ar.odc_dev, ar.adc_dev, olaps_sh, active, ar.iteration_num, ar.dutyCyclePeriod);

	__syncthreads();

	averageActivity(active_sh, avg_act);

	__syncthreads();

	updateBoosts(ar.adc_dev, ar.boosts_dev, avg_act, ar.boostStrength);

	bumpUpColumnsWithWeakOdc(ar.odc_dev, ar.per_dev, ar.numPot_dev, ar.minOdc, ar.synPermBelowStimulusInc, ar.MAX_CONNECTED);

	if(ar.iteration_num % ar.update_period == 0)
		updateMinOdc(ar.odc_dev, ar.minOdc, ar.minPctOdc, ar.SP_SIZE);
}
