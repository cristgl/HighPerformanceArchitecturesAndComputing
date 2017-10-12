//=============================================================================
// FILE:   mytoy.cu
// AUTHORS: Raul Segura & Manuel Ujaldon (copyright 2014)
// Look for the string "MU" whenever Manuel suggests you to introduce changes
// Feel free to change some other parts of the code too (at your own risk)
//=============================================================================

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "io.h"

//=============================================================================
// CUDA functions.
//=============================================================================

//Error handler for CUDA functions.
void cudaErrorHandler(cudaError_t error, const int LINE)
{
    if (error != cudaSuccess) {
        fprintf(stdout, "ERROR(%d): %s\n", LINE, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

//-----------------------------------------------------------------------------
// Set the GPU device and get its properties.
void getDeviceProperties(const int devID, cudaDeviceProp *deviceProp)
{
    // Set device.
    cudaErrorHandler(cudaSetDevice(devID), __LINE__);

    // Get device properties.
    fprintf(stdout, "Leyendo propiedades del dispositivo %d...\n", devID);
    cudaErrorHandler(cudaGetDeviceProperties(deviceProp, devID), __LINE__);

    fprintf(stdout, "GPU Device %d: \"%s\": capacidad de cómputo %d.%d.\n\n",
            devID, deviceProp->name, deviceProp->major, deviceProp->minor);
}

//=============================================================================
// IOHB functions (Input/Output Harwell-Boeing) adapted from the HB library
//=============================================================================

// Read the input matrix.
void readInputMatrix(const char *matrixFile, int *nrow, int *ncol, int *nnzero,
                     int **colptr, int **rowind, double **values)
{
    // Read the Harwell-Boeing format matrix file.
    fprintf(stdout, "Reading input matrix from %s...\n", matrixFile);
    readHB_newmat_double(matrixFile, nrow, ncol, nnzero,
                         colptr, rowind, values);

    fprintf(stdout, "Matrix in file %s is %d x %d ", matrixFile, *nrow, *ncol);
    fprintf(stdout, "with %d nonzero elements.\n\n", *nnzero);
}

//-----------------------------------------------------------------------------
// Write the output matrix.
void writeOutputMatrix(const char *matrixFile, int nrow, int ncol, int nnzero,
                       int *colptr, int *rowind, double *values)
{
    double *rhs = 0, *guess = 0, *exact = 0;
    char mxtype[] = "RUA";
    char ptrfmt[] = "(10I8)";
    char indfmt[] = "(10I8)";
    char valfmt[] = "(5E16.8)";
    char rhsfmt[] = "(5E16.8)";

    // Write the results of your computation into a file named "eureka",
    // which follows the Harwell-Boeing format.
    // POINT 1: Puedes cambiar el nombre "Eureka" si quieres comparar dos versiones de código diferentes.
    // O en caso de que quieras estar seguro de que algunas ejecuciones del mismo código producen exactamente el mismo resultado  (no race conditions occur when your
    // parallel strategy is deployed).
    //
    // Incluso podrías evitar llamar a esta función si la operación de salida es demasiado larga.
    fprintf(stdout, "Writing output matrix in %s...\n", matrixFile);
    writeHB_mat_double(matrixFile, nrow, ncol, nnzero, colptr, rowind, values,
                       0, rhs, guess, exact, matrixFile, "eureka", mxtype,
                       ptrfmt, indfmt, valfmt, rhsfmt, "FGN");

    fprintf(stdout, "Generated file %s successfully.\n\n", matrixFile);
}

//=============================================================================
// The CUDA Kernel.
//=============================================================================

// Cada hebra añade el elemento que le ha sido asignado a la matriz dispersa

// POINT 2: Cambia el tipo de dato a int, float or double
// You may want to change "float *dvalues" by "double *dvalues" in case
// you are curious to see how much GFLOPS drop when using double precision.
// Or even use "int dvalues" if you want to measure performance in integer ALUs.
// (see also hint MU4 below)
__global__ void kernelAdd(float *dvalues, int numOperations,
                          int firstInd, int nextColInd)
{
    int vi = firstInd + blockIdx.x * blockDim.x + threadIdx.x;

// "numOperations" is the 2nd input parameter to our executable
    if (vi < nextColInd) {
        for (int j=0; j<numOperations; ++j) {
            // The operation performed on each nonzero of our sparse matrix:
            dvalues[vi] *=dvalues[vi]+dvalues[vi]*dvalues[vi]; // POINT 3: Choices you may try here:
        }                               // *= (for multiply), /= (for division),
    }                                   // or you may investigate some other :-)
}

//=============================================================================
// Main.
//=============================================================================

int main(int argc, char **argv)
{
	// =======================   Declaración de variables     ==================
	//=========================================================================
    // Variables.
    // CUDA.
    cudaDeviceProp deviceProp;
    cudaStream_t *stream;
    cudaEvent_t start, stop;

    // Matrix.
    // Harwell-Boeing format.
    int nrow, ncol, nnzero;
    // Compressed Sparse Column format.
    int *colptr, *rowind;
    float *values;  // POINT 4: Puedes usar int para medir el rendimeinto en operaciones en punto fijo
    // o double para doble precisión
    double *values64; //

    // To measure time elapsed and performance achieved
    float msecMemHst, msecMemDvc, msecCompStr, msecCompKrn;
    float numOperationsPerValue, numFloatingPointOperations, opIntensity;
    double flops, gigaFlops;

    // Misc.
    int devID;
    int *blocks;
    int *threads;
    float *dvalues;  // POINT 5: This declaration is binded to hints MU2 and MU4

    // ======================= Comprobación de parámetros de entrada ==================
    //=========================================================================
    // Check command line arguments.
    if (argc < 5) {
        fprintf(stderr, "ERROR: Número equivocado de argumentos: %d\n", argc - 1);
        fprintf(stderr, "Use: ./mytoy <deviceID> <numOperationsPer");
        fprintf(stderr, "Value> <inputMatrixFile> <outputMatrixFile>\n");
        exit(EXIT_FAILURE);
    }

     //-------------------------------------------------------------------------
    // This part is just to restrict the execution to device (GPU) 0 or 1
    devID = atoi(argv[1]);
    if ((devID != 0) && (devID != 1)) {
        fprintf(stderr, "ERROR: El primero parámetro es   %s.\n", argv[1]);
        fprintf(stderr, "Tiene que ser 0 para seleccionar el dispositivo GPU en el que vamos a ejecutar.");
        exit(EXIT_FAILURE);
    }

    numOperationsPerValue = atoi(argv[2]);
    if (numOperationsPerValue <= 0) {
        fprintf(stderr, "ERROR: El segundo parámetro es incorrecto: %s.\n", argv[2]);
        fprintf(stderr, "Representa el número de operaciones por valor y debe ser mayor que 0 ");
        exit(EXIT_FAILURE);
    }

    // ======================= Lectura de las características de la tarjeta ==================
    //=========================================================================
    // Get properties of the chosen device.
    getDeviceProperties(devID, &deviceProp);

    // =================== Creación de eventos para monitorizar el tiempo ========
    //-------------------------------------------------------------------------
    // Create CUDA events for timing.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //======================  Lectura de la matriz de entrada ===================================================
    // Lee la matriz de entrada.
    readInputMatrix(argv[3], &nrow, &ncol, &nnzero,
                    &colptr, &rowind, &values64);
    fprintf(stderr,"Tamaño de la matriz, nrow=%d, ncol=%d\n",nrow,ncol);

    // ======================= Reserva de memoria ==================
    //  POINT 6: Aquí hay que especificar el tipo de dato que puede ser float, double o int (ver Punto 2, punto 4 y punto 5)
    values = (float*)malloc(nnzero * sizeof(float));
    for (int i=0; i<nnzero; ++i) {
    //  POINT 7: No olvides cambiar el casting según la declaración del punto 2, 4, 5 y 6
        values[i] = (float)values64[i];
    }

    // ======================= Valores para calcular los bloques y el número de hebras por bloque  ==================

    // Maximum number of threads per block and warp size.
    int maxThreadsPerBlock = 1024;
    const int warpSize = 32;  // Esto no se puede cambiar, no es optativo.

    // ======================= Calculo del grid de hebras ==================

    // Calcular el número de bloques y de hebras que necesitamos para cada columna
    // POINT 8: Aquí tienes que establecer el tamaño del grid para sacarle todo el paralelismo que puedas al lanzamiento del kernel (Ver punto 11)
    blocks = (int*)malloc(ncol * sizeof(float));
    threads = (int*)malloc(ncol * sizeof(float));

    // ======================= Cálculo de los diferentes grids que vamos a lanzar  ==================

    for (int i=0; i<ncol; ++i) {
        fprintf(stderr,"Para i=%d, (((colptr[%d](%d) - colptr[%d](%d))/%d)+1)*%d",i, i+1,colptr[i+1],i,colptr[i],warpSize,warpSize);

        threads[i] = (((colptr[i+1] - colptr[i]) / warpSize) + 1) * warpSize;
        fprintf(stderr,"->>>> threads[%d]=%d",i,threads[i]);

        if (threads[i] <= maxThreadsPerBlock) {
            blocks[i] = 1;
        } else {
            blocks[i] = threads[i] / maxThreadsPerBlock;
            if (threads[i] % maxThreadsPerBlock > 0) {blocks[i]++;}
            threads[i] = maxThreadsPerBlock;
            fprintf(stderr,"->>>> threads[%d]=%d",i,threads[i]);

        }
        fprintf(stderr,"->>>> Blocks[%d]=%d\n",i,blocks[i]);
    }


    //=========================  Ejecución de los grids ===================================
    // Ejecución
    fprintf(stdout, "Running mytoy.\n");

    //-------------------------------------------------------------------------
    // Copy matrix values from host memory to device memory.

    //PUNTO  9: Hay que adecuar el tipo de dato a float, int o double (Ver puntos 2, 4, 5, 6 y 7)
    int valuesSize = nnzero * sizeof(float);

    cudaErrorHandler(cudaEventRecord(start, NULL), __LINE__);

    //fprintf(stdout, "Reservando %d bytes en la memoria del ", valuesSize);
    //fprintf(stdout, "dispositivo para los valores del array ...\n");
    cudaErrorHandler(cudaMalloc((void**)&dvalues, valuesSize), __LINE__);

    //fprintf(stdout, "Copiando datos desde la memoria del host hasta la memoria del dispositivo...\n");
    cudaErrorHandler(cudaMemcpy(dvalues, values, valuesSize,
                                cudaMemcpyHostToDevice), __LINE__);

    cudaErrorHandler(cudaEventRecord(stop, NULL), __LINE__); // Registra el momento del evento de finalización de la copia de la memoria
    cudaErrorHandler(cudaEventSynchronize(stop), __LINE__);
    cudaErrorHandler(cudaEventElapsedTime(&msecMemHst, start, stop), __LINE__); // Calcula el tiempo transcurridos con una precisión de 0.5 microsegundos

    //-------------------------------------------------------------------------
    // Create streams.
    cudaErrorHandler(cudaEventRecord(start, NULL), __LINE__); // Comienza el siguiente tramo de código

    // 	PUNTO 10: Si crees que un sólo stream es mejor para toda la matriz,
    // sólo tienes que reemplazar la siguiente sentencia y el bucle por la siguiente línea
     // cudaErrorHandler(cudaStreamCreate(&stream), __LINE__);

    stream = (cudaStream_t*)malloc(ncol * sizeof(cudaStream_t));
    for (int i=0; i<ncol; ++i) {
        cudaErrorHandler(cudaStreamCreate(&stream[i]), __LINE__);
    }
    //fprintf(stdout, "Stream(s) Creado correctamente.\n");

    cudaErrorHandler(cudaEventRecord(stop, NULL), __LINE__); // Registra la finalización del evento
    cudaErrorHandler(cudaEventSynchronize(stop), __LINE__); // Sincroniza
    cudaErrorHandler(cudaEventElapsedTime(&msecCompStr, start, stop),__LINE__); // Calcula el tiempo

    //-------------------------------------------------------------------------
    // Launch streams.
    cudaErrorHandler(cudaEventRecord(start, NULL), __LINE__); // Comienza el lanzamiento

    //fprintf(stdout, "Lanzando un stream por columna...\n");
    for (int i=0; i<ncol; ++i) { // PUNTO 11: La forma en la que se despliega el paralelismo está aquí.
    	// Reemplaza stream[i] por stream en la siguiente línea si has hecho el cambio del punto 9
        kernelAdd<<<blocks[i], threads[i], 0, stream>>>(dvalues, numOperationsPerValue, colptr[i], colptr[i+1]);
    }
    //fprintf(stdout, "Ejecutando los streams...\n");

    cudaErrorHandler(cudaEventRecord(stop, NULL), __LINE__);
    cudaErrorHandler(cudaEventSynchronize(stop), __LINE__);
    cudaErrorHandler(cudaEventElapsedTime(&msecCompKrn, start, stop),__LINE__);

    cudaErrorHandler(cudaDeviceSynchronize(), __LINE__);
    fprintf(stdout, "Streams executed successfully.\n");

    //-------------------------------------------------------------------------
    // Copiar los resultados de vuelta a la CPU.
    cudaErrorHandler(cudaEventRecord(start, NULL), __LINE__);

    fprintf(stdout, "Copiando los valores de vuelta desde la... ");
    fprintf(stdout, "memoria del dispositivo hasta la memoria del host...\n\n");
    cudaErrorHandler(cudaMemcpy(values, dvalues, valuesSize,
                                cudaMemcpyDeviceToHost), __LINE__);

    cudaErrorHandler(cudaEventRecord(stop, NULL), __LINE__);
    cudaErrorHandler(cudaEventSynchronize(stop), __LINE__);
    cudaErrorHandler(cudaEventElapsedTime(&msecMemDvc, start, stop), __LINE__);

    //=======================Escribir matriz de salida ======================
    // Escribir la matriz de salida
    for (int i=0; i<nnzero; ++i) {
        values64[i] = (double)values[i];
    }

    writeOutputMatrix(argv[4], nrow, ncol, nnzero,
                      colptr, rowind, values64);


    // ======================= Calculo de rendimiento ==================

    // Imprimiendo tiempos y porcentages.
    float msecMem = msecMemHst + msecMemDvc;
    float msecComp = msecCompStr + msecCompKrn;
    fprintf(stdout, "Tiempo de acceso a la memoria de la GPU: %.4f ms.\n\n", msecMem);

    fprintf(stdout, "Creación de streams en la GPU:  %.4f ms.\n", msecCompStr);
    fprintf(stdout, "Tiempo de ejecución del kernel: %.4f ms.\n", msecCompKrn);
    fprintf(stdout, "Tiempo de computación en GPU:   %.4f ms.\n\n", msecComp);

    //PUNTO 12:	Cambia float, int or double según el punto 2, 4, 5, 6, 7 y 8
    opIntensity = numOperationsPerValue / sizeof(float);
    fprintf(stdout, "Operaciones en punto flotante por byte: %.4f FLOP/byte.\n", opIntensity);

    numFloatingPointOperations = nnzero * numOperationsPerValue;
    flops = numFloatingPointOperations / (msecComp / 1000.0f);
    gigaFlops = flops * 1.0e-9f;
    fprintf(stdout, "Rendimiento: %.4f GFLOP/s.\n\n", gigaFlops);

    //=========================================================================
    // Free host memory.
    free(colptr); free(rowind); free(values);
    free(blocks); free(threads);

    // liberación.
    cudaErrorHandler(cudaDeviceReset(), __LINE__);

    return EXIT_SUCCESS;
}
