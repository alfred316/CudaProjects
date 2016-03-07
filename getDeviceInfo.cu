//Alfred Shaker
//10/23/2015
//Homework 2

#include <stdio.h>

//function to get and print device properties
void printDeviceProperties(cudaDeviceProp devProp)
{
	//get the cuda driver version
	int driverVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	printf("Version Number:  %d\n",driverVersion/1000 );
	//get the number of multiprocessors
	int mp = devProp.multiProcessorCount;
	printf("Number of MultipPrcessors:  %d\n", mp);
	//check the computation capacity and calculate cores based on that
	int cores = 0;
	switch(devProp.major)
	{
	  //Fermi
	  case 2:
	    if (devProp.minor == 1){ cores = mp * 48;
		printf("Number of cores: %d\n", cores);
	    }
	    else{ cores = mp * 32;
		printf("Number of cores: %d\n", cores);
	    }
	    break;
	  //Kepler
	  case 3: 
	    cores = mp * 192;
	    printf("Number of cores: %d\n", cores);
	    break;
	  //Maxwell
	  case 5:
	    cores = mp * 128;
	    printf("Number of cores: %d\n", cores);
	    break;
	  default:
	    printf("Unknown device type\n");
	    break;
	}
	int kb = 1024;
	int mb = kb * kb;
	//get the total global memory in megabytes
	int globalMemory = devProp.totalGlobalMem / mb;
	printf("Total Global Memory: %d mb\n", globalMemory);
	//get the shared memory per block in kilobytes
	int sharedMemory = devProp.sharedMemPerBlock / kb;
	printf("Shared Memory Per Block: %d kb\n", sharedMemory);
	//get the maximum number of threads per block
	int maxThreads = devProp.maxThreadsPerBlock;
	printf("Maximum Threads Per Block: %d\n", maxThreads);
	//get the maximum dimension size of the block
	int maxBlockDim = devProp.maxThreadsDim[0];
	printf("Maximum Size of Block Dimensions: %d\n", maxBlockDim);
	//get the maximum dimension size for the grid
	int maxGridDim = devProp.maxGridSize[0];
	printf("Maximum Size of Grid Dimensions: %d\n", maxGridDim);
 
}

int main()
{
	//get number of devices
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Total number of Devices: %d\n", deviceCount);
	//iterate through each device and get properties of each one
	for(int i = 0; i < deviceCount; ++i)
	{
	  printf("Showing info for device number %d\n", i+1);
	  cudaDeviceProp devProp;
	  cudaGetDeviceProperties(&devProp, i);
	  printDeviceProperties(devProp);	
	}
        return 0;
}
