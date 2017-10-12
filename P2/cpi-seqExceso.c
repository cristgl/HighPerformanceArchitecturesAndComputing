#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "time.h"




main(int argc, char **argv)
{
  register double width, sum;
  register int intervals, i;
  double c1,c2;

  /* get the number of intervals */
  intervals = atoi(argv[1]);
  c1 = MPI_Wtime();
  width = 1.0 / intervals;

  /* do the computation */
  sum = 0;
  for (i=0; i<intervals; ++i) {
	register double x = (i + 1.0) * width;
	sum += 4.0 / (1.0 + x * x);
  }
  sum *= width;

  register double enu = fabs(3.141592653589793238462643 - sum);
  register double error = enu/3.141592653589793238462643*100;
   c2 = MPI_Wtime();
	
  printf("Estimation of pi is %.14f\n", sum);
  printf("Percentage error of pi is %.14f\n", error);
  printf("Time: %.14f\n", (c2-c1)*1000/CLOCKS_PER_SEC); 

  return(0);
}

