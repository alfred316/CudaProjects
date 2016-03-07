//Alfred Shaker
//10-13-2015

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 

// CUDA kernel
__global__ void vectorSum(int *a, int *b, int *c, int n)
{
	//get the id of global thread
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	
	//checks to make sure we're not out of bounds
	if(id < n) 
           c[id] = a[id] + b[id];

}

int main(int argc, char* argv[])
{
	//size of vectors
	int size = 100;
	//host side vectors
	int *h_a, *h_b, *h_c;
	//device side vectors
	int *d_a, *d_b, *d_c;
	
	//size of each vector in bytes
	size_t bytes = size*sizeof(int); 
	
	//allocate memory for host side vectors
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	//allocate memory for device side vectors
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	//initiate the vectors with random numbers between 0 and 9
	int i;
	for(i = 0; i < size; i++)
	{
	  h_a[i] = rand() % 10;
	  h_b[i] = rand() % 10;
	}
	
	//copy host vectors to device vectors
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	//number of threads in each block
	int blockSize = 1024;
	//number of thread blocks in grid
	int gridSize = (int)ceil((float)size/blockSize);
	
	//execute kernel function
	vectorSum<<<gridSize, blockSize >>>(d_a, d_b, d_c, size);
	
	//copy the vector back to the host from the device only for the result vector
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
	
	//print out each result, preceeding it with the values of the added vectors for that index
	for(i = 0; i < size; i++)
		printf("a: %d, b: %d, c[%d] = %d\n",h_a[i], h_b[i], i, h_c[i] );

	//release the device side memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//release the host side memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
