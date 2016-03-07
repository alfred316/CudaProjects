//Alfred Shaker
//Octtober 30th 2015
//Matrix Multiply

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//CUDA kernel function
__global__ void matrixMultiply(float* a, float* b, float* c, int n)
{
	//use block dimentions to calculate column and row
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	
	for(int i = 0; i<n; i++)
	{
	  c[row*n + col]+= a[row*n + i] + b[i*n + col];
	}
}


//main function
int main(int argc, char* argv[])
{
	//declare base values and 2d arrays
	int width = 6;
	float A_h[width][width], B_h[width][width], C_h[width][width];
	float *A_d, *B_d, *C_d;
	
	int tileWidth = 2;
	int i, j;
	
	//fill up arrays with random values between 0 and 9
	for (i = 0; i < width; i++)
	{
	    for( j = 0; j < width; j++)
	    {
	      A_h[i][j] = rand() % 10;
	      B_h[i][j] = rand() % 10;
	    }
	}
	
	//allocate memory for device variables
	cudaMalloc((void**)&A_d, width*width*sizeof(int));
	cudaMalloc((void**)&B_d, width*width*sizeof(int));
	
	//copy values from host variables to device variables
	cudaMemcpy(A_d, A_h, width*width*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, width*width*sizeof(int), cudaMemcpyHostToDevice);

	//allocate memory for the device result array 
	cudaMalloc((void**)&C_d, width*width*sizeof(int));

	//create variables for grid and block dimentions
	dim3 dimGrid(width/tileWidth, width/tileWidth, 1);
	dim3 dimBlock(tileWidth, tileWidth, 1);

	//call the kernel function
	matrixMultiply<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);

	//copy values from the device result array to the host result array
	cudaMemcpy(C_h, C_d, width*width*sizeof(int), cudaMemcpyDeviceToHost);
	
	//print the result of the matrix multiplication
	for(i = 0; i<width; i++)
	{
	    for(j = 0; j<width; j++)
	    {
	      printf("Index: [%d] [%d] Value:[%f] [%f]. Result: %f\n",i, j,A_h[i][j], B_h[i][j],  C_h[i][j]);
	    }
	}
	
	//deallocate device variables memory	
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);


	return 0;	
}
