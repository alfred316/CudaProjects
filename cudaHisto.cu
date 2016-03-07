//Alfred Shaker
//Dec 18th 2015
//CUDA Histogram

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define DIM 512

__global__ void kernelHisto(const char *const buffer, int *histo, int *A, int *B)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int id  = row*DIM + col;
 

    for(int i = 0; i < row; ++i)
      {
	for(int j = 0; j < col; ++j)
	  {
	    A[i][j] = rand() % 201;
	    B[i][j] = rand() % 201;

	    atomicAdd(&(histo[buffer[id]]));
	  }
      }

}

int main()
{
    int histo[DIM*DIM];

    char *buffer;

    int *A;
    int *B;

    cudaMalloc((void**)&A, DIM*DIM*sizeof(int));
    cudaMalloc((void**)&B, DIM*DIM*sizeof(int));

    dim3 dimGrid(DIM, DIM, 1);
    dim3 dimBlock(DIM, DIM, 1);

    kernelHisto<<<dimGrid, dimBlock>>>(buffer, histo, A, B);
   
    cudaFree(A);
    cudaFree(B); 

return 0;
}
