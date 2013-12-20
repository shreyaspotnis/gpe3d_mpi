#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <unistd.h>

#include "ini.h"

#define TRUE 1
#define FALSE 0

#define MASTER_RANK 0
#define NX  256
#define NY  16
#define NZ  16


typedef struct configuration {
    int Nx;
    int Ny;
    int Nz;
    double dt;
} configuration;

/* Globals */
int rank, size;
/* End of globals */

/* Function declarations */
int create_plans(fftw_plan *p_fwd, fftw_plan *p_bwd, fftw_complex *psi_local);
int read_config(configuration *cfg, int argc, char **argv);
static int handler(void* user, const char* section, const char* name,
                   const char* value);
/* End of Function declarations */


int main(int argc, char **argv) {

    configuration cfg;

    // Initialize MPI and fftw for MPI
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    read_config(&cfg, argc, argv);

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
    
    printf("task %d/%d!. x_s:%d Nx:%d\n", rank, size, x_start_local, cfg.Nx);


    // clean up
    fftw_destroy_plan(p_fwd);
    fftw_destroy_plan(p_bwd);
    fftw_free(psi_local);
    MPI_Finalize();

    return 0;
}

int read_config(configuration *cfg, int argc, char **argv) {
    char *input_filename = NULL;
    // read inputs from file
    if(rank == MASTER_RANK) {
        int success = FALSE;
        int c;
        FILE *fp;
        success = FALSE;
        while ((c = getopt(argc, argv, "f:")) != EOF) {
            switch(c) {
            case 'f':
                input_filename = optarg;
                success = TRUE;
                break;
            } // end of switch(c)
        } // end of while
        
        if (ini_parse(input_filename, handler, cfg) < 0) {
                printf("Can't load 'test.ini'\n");
                success = FALSE;
        }
        if(!success)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // send configuration to all processors
    MPI_Bcast(cfg, sizeof(configuration), MPI_CHAR, MASTER_RANK,
              MPI_COMM_WORLD);
}

int create_plans(fftw_plan *p_fwd, fftw_plan *p_bwd, fftw_complex *psi_local) {

    *p_fwd = fftw_mpi_plan_dft_3d(NX, NY, NZ, psi_local, psi_local,
                                    MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    *p_bwd = fftw_mpi_plan_dft_3d(NX, NY, NZ, psi_local, psi_local,
                                     MPI_COMM_WORLD, FFTW_BACKWARD,
                                     FFTW_MEASURE);
    return 0;
}

static int handler(void* user, const char* section, const char* name,
                   const char* value) {
    configuration* pconfig = (configuration*)user;

    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0
    if (MATCH("sim", "Nx")) 
        pconfig->Nx = atoi(value);
    else if (MATCH("sim", "Ny")) 
        pconfig->Ny = atoi(value);
    else if (MATCH("sim", "Nz")) 
        pconfig->Nz = atoi(value);
    else 
        return 0;  /* unknown section/name, error */
    return 1;
}
