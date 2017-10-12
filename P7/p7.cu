#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#define BILLION 1E9;


const int n=300;

__global__ void grayscaleKernel(int *ms, int *aux, int n){
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	int k=0;

	int grayscale=0;
	if(i<n){
		for(k=0; k<n-3; k+=3){
			grayscale = 0.299*ms[i*n+k] + 0.5876*ms[i*n+k+1] + 0.114*ms[i*n+k+2];
			aux[i*n+k] = aux[i*n+k+1] = aux[i*n+k+2] = grayscale;
		}
	}
}

int main(){
	int m[n][n], a[n][n];
	int *d_m, *d_a;
	struct timespec requestStart, requestEnd, reqSt, reqEd;
	static unsigned int color[3];

	FILE *fp = fopen("picture_col.ppm", "w");
	(void) fprintf(fp,"P6\n%d %d\n255\n", n, n);

	FILE *sp = fopen("picture_gray.ppm", "w");
	(void) fprintf(sp,"P6\n%d %d\n255\n", n, n);

	// Inicialización y construcción de imagen a color
	for(int i=0; i<n; i++)
		for(int j=0; j<n-3; j+=3){
			m[i][j] = j%256;
			m[i][j+1] = i%256;
			m[i][j+2] = i*j%256;

			color[0] = m[i][j];
			color[1] = m[i][j+1];
			color[2] = m[i][j+2];
			(void) fwrite(color, sizeof(int), 3, fp);
		}
	int size = sizeof(int)*n*n;

	clock_gettime(CLOCK_REALTIME, &reqSt);
	cudaMalloc((void **) &d_m, size);
	cudaMalloc((void **) &d_a, size);
	clock_gettime(CLOCK_REALTIME, &reqEd);

	cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(n/4, n/4);
	dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);

	clock_gettime(CLOCK_REALTIME, &requestStart);
	grayscaleKernel<<<dimBlock, dimGrid>>>(d_m, d_a, n);
	clock_gettime(CLOCK_REALTIME, &requestEnd);

	cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
	printf("%d %d\n", a[0][0], a[n-1][n-1]);

	cudaFree(d_a);
	cudaFree(d_m);

	for(int i=0; i<n; i++)
		for(int j=0; j<n-3; j+=3){
		    color[0] = a[i][j];
		   	color[1] = a[i][j+1];
		   	color[2] = a[i][j+2];
		   	(void) fwrite(color, sizeof(int), 3, sp);
		}

	double accum = (double) (requestEnd.tv_sec - requestStart.tv_sec) + (requestEnd.tv_nsec - requestStart.tv_nsec)/BILLION;
    printf( "Tiempo de ejecución: %.15lf\n", accum );
    double accum2 = (double) (reqEd.tv_sec - reqSt.tv_sec) + (reqEd.tv_nsec - reqSt.tv_nsec)/BILLION;
    printf( "Tiempo empleado en la reserva de memoria: %.15lf\n", accum2 );

	return cudaThreadExit();	
}