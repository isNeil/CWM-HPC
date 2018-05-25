#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>


//kernel 

__global__ void   


int main() {
	
    int N=100000000;
    int area=0;

    #pragma omp parallel for shared(area)
    for(int i=0; i<N; i++) {
        float x = ((float)rand())/RAND_MAX;
        float y = ((float)rand())/RAND_MAX;
	#pragma omp critical
        if(x*x + y*y <= 1.0f) area++;
    }

    printf("\nPi:\t%f\n", (4.0*area)/(float)N);
/*
    area = 0;	   
    for(int i=0; i<N; i++) {
        double x = ((double)rand())/RAND_MAX;
        double y = ((double)rand())/RAND_MAX;
	if(x*x + y*y <= 1.0) area++;
    }

    printf("\nPi:\t%f\n", (4.0*area)/(double)N);
*/
    return(0);
}
