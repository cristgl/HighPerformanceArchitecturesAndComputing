#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include <unistd.h>
#include <cmath>
#include <time.h> 
#include "mpi.h"
#include <fstream>
using namespace std;

struct matriz{
	int r;
	int g;
	int b;
};

const int n=340;
matriz ms[n][n], aux[n][n];

int main(int argc, char **argv){
	ofstream col_img("picture_col.ppm");
	ofstream gray_img("picture_gray.ppm");
	int size, rank;
	
	double cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8;

	static unsigned int color[3];

	FILE *fp = fopen("picture_col.ppm", "w");
	(void) fprintf(fp, "P6\n%d %d\n255\n", n, n);
	
	FILE *sp = fopen("picture_gray.ppm", "w");
	(void) fprintf(sp, "P6\n%d %d\n255\n", n, n);

	/*
	col_img << "P3" << endl;
	col_img << n << " " << n << endl;
	col_img << "255" << endl;
	gray_img << "P3" << endl;
	gray_img << n << " " << n << endl;
	gray_img << "255" << endl;
	*/

	// Inicialización y construcción de imagen a color
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++){
			ms[i][j].r = j%256;//rand() % 256;
			ms[i][j].g = i%256;//rand() % 256;
			ms[i][j].b = i*j%256;//rand() % 256;

			aux[i][j].r = 0;
			aux[i][j].g = 0;
			aux[i][j].b = 0;

			color[0] = ms[i][j].r;
			color[1] = ms[i][j].g;
			color[2] = ms[i][j].b;
			(void) fwrite(color, sizeof(int), 3, fp);
			//col_img << ms[i][j].r << " " << ms[i][j].g << " " << ms[i][j].b << endl;
		}

   	cl1 = MPI_Wtime();
	MPI_Init(&argc,&argv);
	cl2 = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &size);  

	if(size < 1){
		printf("Need at least 1 process.\n");
		MPI_Finalize();
		return(1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Escala de grises
   	int grayscale=0;
   	int sendto = rand() % size;
   	int i=0;
   	cl3 = MPI_Wtime();
   	for(i=0; i<n-size+1; i++){
   		for(int j=0; j<n; j++){
	   			if(rank==0){
	   				for(int k=0; k<size; k++){
	   					MPI_Send(&ms[i][j+k].r, 1, MPI_INT, rank+k, 0, MPI_COMM_WORLD);
	   					MPI_Send(&ms[i][j+k].g, 1, MPI_INT, rank+k, 1, MPI_COMM_WORLD);
	   					MPI_Send(&ms[i][j+k].b, 1, MPI_INT, rank+k, 2, MPI_COMM_WORLD);
	   				}
	   			}

	   			cl7 = MPI_Wtime();
	   			MPI_Recv(&ms[i][j].r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	   			MPI_Recv(&ms[i][j].g, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	   			MPI_Recv(&ms[i][j].b, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	   			grayscale = 0.299*ms[i][j].r + 0.5876*ms[i][j].g + 0.114*ms[i][j].b;
	   			aux[i][j].r = aux[i][j].g = aux[i][j].b = grayscale;

	   			MPI_Send(&aux[i][j].r,1,MPI_INT,0,rank+100,MPI_COMM_WORLD);
	   			MPI_Send(&aux[i][j].g,1,MPI_INT,0,rank+101,MPI_COMM_WORLD);
	   			MPI_Send(&aux[i][j].b,1,MPI_INT,0,rank+102,MPI_COMM_WORLD);
	   			cl8 = MPI_Wtime();
   				
	   			if(rank==0){
	   				for(int k=0; k<size; k++){
	   					MPI_Recv(&aux[i][j+k].r,1,MPI_INT,MPI_ANY_SOURCE,k+100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	   					MPI_Recv(&aux[i][j+k].g,1,MPI_INT,MPI_ANY_SOURCE,k+101,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	   					MPI_Recv(&aux[i][j+k].b,1,MPI_INT,MPI_ANY_SOURCE,k+102,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	   				}
	   			}
   		}
   	}

   	if (rank==0)
	   	for(int i=0; i<n; i++){
	   		for(int j=0; j<n; j++){
	   			ms[i][j].r=aux[i][j].r;
	   			ms[i][j].g=aux[i][j].g;
	   			ms[i][j].b=aux[i][j].b;
	   		}
	   	}

   	cl4 = MPI_Wtime();

   	// Imagen escala de grises
   	if(rank==0)
	   	for(int i=0; i<n; i++)
	   		for(int j=0; j<n; j++){
	   			color[0] = ms[i][j].r;
	   			color[1] = ms[i][j].g;
	   			color[2] = ms[i][j].b;
	   			(void) fwrite(color, sizeof(int), 3, sp);
	   			//gray_img << ms[i][j].r << " " << ms[i][j].g << " " << ms[i][j].b << endl;
	   		}
	   		

   	cl5 = MPI_Wtime();
   	MPI_Finalize();
   	cl6 = MPI_Wtime();


   	if(rank==0){
   		printf("Tiempo de creación de datos %.15f\n", (cl2-cl1)*1000/CLOCKS_PER_SEC);
   		printf("Tiempo de cómputo incluyendo envío y recepción de datos %.15f\n", (cl4-cl3)*1000/CLOCKS_PER_SEC); 
   		printf("Tiempo de cómputo %.15f\n", (cl8-cl7)*1000/CLOCKS_PER_SEC);
		printf("Tiempo de destrucción de datos %.15f\n", (cl6-cl5)*1000/CLOCKS_PER_SEC);
   	}

	return 0;			
}
