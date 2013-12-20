#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <unistd.h>

#define TRUE 1
#define FALSE 0

#define MASTER_RANK 0
#define NX  256
#define NY  16
#define NZ  16

int create_plans(fftw_plan *p_fwd, fftw_plan *p_bwd, fftw_complex *psi_local);

int main(int argc, char **argv) {

    int rank, size;
    char *input_filename = NULL;

    // Initialize MPI and fftw for MPI
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // read inputs from file
    int success = FALSE;
    #define UNBLOCKME
    #ifdef UNBLOCKME
    if(rank == MASTER_RANK) {
        int c;
        FILE *fp;
        while ((c = getopt(argc, argv, "f:h")) != EOF) {
            switch(c) {
            case 'f':
                input_filename = optarg;
                success = TRUE;
                break;
            case 'h':
                printf("Supply a filename using -f input_filename.txt");
                break;
            } // end of switch(c)
        } // end of while

        if(success) {
            printf("%s", input_filename);
            fp = fopen(input_filename, "r");
            int test_int;
            fscanf(fp, "%d", &test_int);
            printf("%d", test_int);
            fclose(fp);
        }
    }
    #endif

    if(!success) MPI_Abort(MPI_COMM_WORLD, 1);

    // The MPI implementation of fftw splits up the 3d grid into blocks
    // where each node has only a subsection of grid in the X direction
    // and the entire grid in the other two dimensions. Here NX_local will give
    // us the number of elements in the x direction on our local process and
    // x_start_local will give us the start index on the local process.
    // alloc_local gives the amount of memory we need to allocate on the local
    // process.
    ptrdiff_t alloc_local, NX_local, x_start_local; 

    alloc_local = fftw_mpi_local_size_3d(NX, NY, NZ, MPI_COMM_WORLD,
                                         &NX_local, &x_start_local);
    fftw_complex *psi_local;
    fftw_plan p_fwd, p_bwd;
    
    psi_local = fftw_alloc_complex(alloc_local);
    create_plans(&p_fwd, &p_bwd, psi_local);
    
    printf("task %d/%d!. x_s:%d\n", rank, size, x_start_local);


    // clean up
    fftw_destroy_plan(p_fwd);
    fftw_destroy_plan(p_bwd);
    fftw_free(psi_local);
    MPI_Finalize();

    return 0;
}

int create_plans(fftw_plan *p_fwd, fftw_plan *p_bwd, fftw_complex *psi_local) {

    *p_fwd = fftw_mpi_plan_dft_3d(NX, NY, NZ, psi_local, psi_local,
                                    MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    *p_bwd = fftw_mpi_plan_dft_3d(NX, NY, NZ, psi_local, psi_local,
                                     MPI_COMM_WORLD, FFTW_BACKWARD,
                                     FFTW_MEASURE);
    return 0;
}
