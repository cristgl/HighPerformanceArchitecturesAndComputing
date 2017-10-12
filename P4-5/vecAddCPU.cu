#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include cuda.h

using namespace std;

#define BILLION 1E9;

__global__ void vecAddKernel(float *A, float *B, float *C, int n){
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	if(i<n) C[i] = A[i]+B[i];
}

void vecAddK(float *A, float *B, float *C, int n){
	for(int i=0; i<n; i++)
		C[i] = A[i]+B[i];
}

void vecAdd(float *hA, float *hB, float *hC, int n){
struct timespec requestStart, requestEnd, requestS, requestE;
        int size = n*sizeof(float);
        float dA[size], dB[size], dC[size];

        clock_gettime(CLOCK_REALTIME, &requestStart);
        vecAddK(hA,hB,dC,n);
        cout << dC[0] <<" " << dC[n-1] << endl;
        clock_gettime(CLOCK_REALTIME, &requestEnd);

        cudaMalloc((void **) &dA, size);
        cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
        cudaMalloc((void **) &dB, size);
        cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
        cudaMalloc((void **) &dC, size);

		dim3 DimGrid(((n-1)/256)+1,1,1);
        dim3 DimBlock(256,1,1);
        clock_gettime(CLOCK_REALTIME, &requestS);
        vecAddKernel<<<DimGrid,DimBlock>>>(dA,dB,dC,n);
        cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
        cout << dC[0] <<" " << dC[n-1] << endl;
        clock_gettime(CLOCK_REALTIME, &requestE);

        double accum = (double) (requestEnd.tv_sec - requestStart.tv_sec) + (requestEnd.tv_nsec - requestStart.tv_nsec)/BILLION;
        printf( "CPU %lf\n", accum );
        double accumu = (double) (requestE.tv_sec - requestS.tv_sec) + (requestE.tv_nsec - requestS.tv_nsec) / BILLION;
		printf( "GPU %lf\n", accumu );

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cudaError_t err = cudaMalloc((void **) &dA, size);


	if(err != cudaSuccess){
		printf("%s en %s en línea %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
		exit(EXIT_FAILURE);
	}
}

int main(){
	int n, i=0;
	struct timespec requestStart, requestEnd;
	string line;
	ifstream fA("data/0/input0.raw");
	ifstream fB("data/0/input1.raw");
	ifstream fC("data/0/output.raw");

	if(fA){
		getline(fA,line);
		n = stoi(line,NULL,0);
	}

	float hA[n];
	float hB[n];
	float hC[n];
	//float dC[n];

	if(fA){
		//getline(fA,line);
		//n = stoi(line,NULL,0);

		while(getline(fA,line)){
			float f = stof(line,NULL);
			hA[i] = f;
			i++;
		}

		fA.close();
	}

	i=0;
	if(fB){
		getline(fB,line);
		n = stoi(line,NULL,0);
		while(getline(fB,line)){
			float f = stof(line,NULL);
			hB[i] = f;
			i++;
		}

		fB.close();
	}

	//clock_gettime(CLOCK_REALTIME, &requestStart);
	vecAdd(hA, hB, hC, n);
	//clock_gettime(CLOCK_REALTIME, &requestEnd);

	//double accum = (double) (requestEnd.tv_sec - requestStart.tv_sec) + (requestEnd.tv_nsec - requestStart.tv_nsec) / BILLION;
	//printf( "%lf\n", accum );
/*
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++)
			cout << hC[i][j] << " ";
		cout << endl;
	}
*/
	/*i=0;
	if(fC){
		getline(fC,line);
		n = stoi(line,NULL,0);
		while(getline(fC,line)){
			float f = stof(line,NULL);
			dC[i] = f;
			i++;
		}

		fC.close();
	}

	bool correcto=true;
	for(int i=0; i<n; i++){
		cout << "hC[" << i << "]: " << hC[i] << endl;
		cout << "dC[" << i << "]: " << dC[i] << endl;
		if(abs(dC[i]-hC[i])>1)
			correcto=false;
	}

	cout << "correcto: "<< correcto << endl;*/
}
