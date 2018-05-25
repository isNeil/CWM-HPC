#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>








//GPU kernel add 1 to area if x and y inside circle

// d_ specifies device, h_ specifies host, R is radius of given point

 
__global__ void test_point( int *N, int *area, float *d_x, float *d_y, float *d_R, float *d_area){

	int index = blockDim.x*blockIdx.x + threadIdx.x;
	
	d_R[index] = d_x[index]*d_x[index]+d_y[index]*d_y[index];
	
//sync threads to prevent deadlock	
	
	__syncthreads();

	if(d_R[index] < 1) atomicAdd(&d_area[blockIdx.x],1);

}
 
__global__ void area_reduction(float *d_area){
        
	// allocate shared memory
	extern __shared__ float shared_array[];

	int index = threadIdx.x;
	shared_array[index] = d_area[index];

	for(float d = blockDim.x/2; d>0; d/=2){
		__synchthreads();
		if(index<d){
			shared_array[index] += shared_array[index + d];
		}
	}
	
	if(index == 0) d_area[0] = shared_array[0];
}



int main(void) {

//N is number of random points plotted
//area specifies number of points in quadrant of unit circle

        size_t N=100000000;


//initiate GPU

        int nBlocks = N/256;
        int nThreads =256;

//use GPU with id = 0
        int deviceid= 0;
        int devCount;

//checks if GPU is available

        cudaGetDeviceCount(&devCount);
        if(deviceid<devCount) cudaSetDevice(deviceid);
        else return(1);

// declare rng

        curandGenerator_t gen;


//pointers to device and host memory        
	float *h_x, *h_y, *d_x, *d_y, *d_R, *d_area, *h_area; 


//allocate host memory

        h_x = (float*) malloc(N*sizeof(float));
        h_y = (float*) malloc(N*sizeof(float));
	h_area = (float*)malloc(nBlocks*sizeof(float));


//allocate memory on the device


	cudaMalloc((void **)&d_x, N*sizeof(float));
	cudaMalloc((void **)&d_y, N*sizeof(float));
        cudaMalloc((void **)&d_area, nBlocks*sizeof(float));
        cudaMalloc((void **)&d_R, N*sizeof(float));
	


//create pseudo-RNG


	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);


//set seed

	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

//generate N random numbers

	curandGenerateUniform(gen, d_x, N);

	curandGenerateUniform(gen, d_y, N);
 

//Synchronise device so all blocks finish calculations
	
	cudaDeviceSynchronize();

//run kernal for reducing sum of areas of blocks

	area_reduction<<<nBlocks/nThreads, nThreads, nBlocks*sizeof(float)>>>(d_area);

//transfer results to host

        cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_area, d_area, nBlocks*sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nPi:\t%f\n", (4.0*h_area)/(float)N);
 




//cleanup

	curandDestroyGenerator(gen);
	cudaFree(d_x);
	cudaFree(d_y);
	free(h_y);
	free(h_x);


/*
    area = 0;	   
    for(int i=0; i<N; i++) {
        double x = ((double)rand())/RAND_MAX;
        double y = ((double)rand())/RAND_MAX;
	if(x*x + y*y <= 1.0) h_area++;
    }

    printf("\nPi:\t%f\n", (4.0*area)/(double)N);
*/
	return(0);
}
