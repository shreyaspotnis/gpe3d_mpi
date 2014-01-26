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

/* Preprocessor definitions */
#define TRUE 1
#define FALSE 0
#define HBAR 1.05457173e-34
#define M_RB 1.44e-25
#define A_BG 5.1e-9
#define PI 3.1415926
/* End of preprocessor definitions */

#define MASTER_RANK 0

typedef struct configuration {
    int Nx;
    int Ny;
    int Nz;
    int Nt;
    int Nt_store;
    double dt;
    double dx;

    double time_scale;
    double length_scale;
    double energy_scale;

    double K_mult; // multiplier for the kinetic energy term
    double U_mult; // multiplier for the potential energy term
    double I_mult; // multiplier for the interaction energy term
} configuration;

/* Globals */
int rank, size;
/* End of globals */

/* Function declarations */
int create_plans(configuration *cfg, fftw_plan *p_fwd, fftw_plan *p_bwd,
                 fftw_complex *psi_local);
int read_config(configuration *cfg, int argc, char **argv);
int process_config(configuration *cfg);
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
    // and the entire grid in the other two dimensions. Here Nx_local will give
    // us the number of elements in the x direction on our local process and
    // x_start_local will give us the start index on the local process.
    // alloc_local gives the amount of memory we need to allocate on the local
    // process.
    ptrdiff_t alloc_local, Nx_local, x_start_local; 

    alloc_local = fftw_mpi_local_size_3d(cfg.Nx, cfg.Ny, cfg.Nz, MPI_COMM_WORLD,
                                         &Nx_local, &x_start_local);
    fftw_complex *psi_local;
    fftw_plan p_fwd, p_bwd;
    
    psi_local = fftw_alloc_complex(alloc_local);
    create_plans(&cfg, &p_fwd, &p_bwd, psi_local);
    
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

int create_plans(configuration *cfg, fftw_plan *p_fwd, fftw_plan *p_bwd,
                 fftw_complex *psi_local) {

    *p_fwd = fftw_mpi_plan_dft_3d(cfg->Nx, cfg->Ny, cfg->Nz, psi_local,
                                  psi_local, MPI_COMM_WORLD, FFTW_FORWARD,
                                  FFTW_MEASURE);
    *p_bwd = fftw_mpi_plan_dft_3d(cfg->Nx, cfg->Ny, cfg->Nz, psi_local,
                                  psi_local, MPI_COMM_WORLD, FFTW_BACKWARD,
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
    else if (MATCH("sim", "Nt")) 
        pconfig->Nt = atoi(value);
    else if (MATCH("sim", "Nt_store")) 
        pconfig->Nt_store = atoi(value);
    else if (MATCH("sim", "dt")) 
        pconfig->dt = atof(value);
    else if (MATCH("sim", "dx")) 
        pconfig->dx = atof(value);
    else if (MATCH("sim", "time_scale")) 
        pconfig->time_scale = atof(value);
    else if (MATCH("sim", "length_scale")) 
        pconfig->length_scale = atof(value);
    else 
        return 0;  /* unknown section/name, error */
    return 1;
}

int process_config(configuration *cfg) {
    double ls = cfg->length_scale;
    cfg->energy_scale = HBAR / cfg->time_scale;
    cfg->K_mult = HBAR*HBAR/(2.0 * M_RB * ls * ls)
                  / cfg->energy_scale;

    cfg->U_mult = 0.0;
    // TODO: check this again
    cfg->I_mult = (4.0 * PI * HBAR * HBAR * A_BG)/M_RB/(ls*ls*ls)
                  / cfg->energy_scale;
    return 0;
}
