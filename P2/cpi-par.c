#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "time.h"

main(int argc, char **argv)
{
// double t = MPI_Wtime()
	// /usr/lib64/openmpi/bin/mpicc cpi-par.c -o cpiMaribel
	//echo '/usr/lib64/openmpi/bin/mpiexec -n 2 ./cpiMaribel' |qsub -q acap

  register double width;
  register int intervals, i;
  int   size, rank;
  double  sum, result;
  double cl, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10;

  cl = MPI_Wtime();
  MPI_Init(&argc,&argv);
  cl2 =  MPI_Wtime();

  MPI_Comm_size(MPI_COMM_WORLD, &size);  

  if (size < 1) {
    printf("Need at least 1 processes.\n");
    MPI_Finalize();
    return(1);
  }

  cl3 =  MPI_Wtime();
  intervals = atoi(argv[1]);
  width = 1.0 / intervals;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  sum = 0;
  int j=0;

  for(int i=rank; i<intervals && j<intervals; i+=size){
    register double x = (i + 0.5) * width;
    sum += 4.0 / (1.0 + x * x);
  }
  cl4 =  MPI_Wtime();

  cl7 =  MPI_Wtime();  
  MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  cl8 =  MPI_Wtime();

  if(rank==0){
  	cl5 =  MPI_Wtime();
    result*=width;
    
    register double enu = (3.141592653589793238462643 - result)*(-1);
    register double error = enu/3.141592653589793238462643*100;

    cl6 =  MPI_Wtime();

    printf("Estimation of pi is %.10f\n", result);
    printf("Percentage error of pi is %.10f\n", error);

    printf("Tiempo de creación de datos %.10f\n", (cl2-cl)*1000/CLOCKS_PER_SEC);
    double tpo = (cl4-cl3)*1000/CLOCKS_PER_SEC + (cl6-cl5)*1000/CLOCKS_PER_SEC;
    printf("Tiempo de cálculo %.10f\n", tpo);
    printf("Tiempo de recogida de datos %.10f\n", (cl8-cl7)*1000/CLOCKS_PER_SEC);

  }  

  cl9 =  MPI_Wtime();
  MPI_Finalize();
  cl10 =  MPI_Wtime();

  if(rank==0)
  	printf("Tiempo de destrucción de datos %f\n", (cl10-cl9)*1000/CLOCKS_PER_SEC);

  return(0);
}

