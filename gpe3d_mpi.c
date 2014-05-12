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
#include "common.h"
#include "configuration.h"

/* Globals */
int rank, size;
/* End of globals */


/* Function declarations */
int create_plans(configuration *cfg, fftw_plan *p_fwd, fftw_plan *p_bwd,
                 fftw_complex *psi_local);


int create_1d_grids(double **x_grid, double **y_grid, double **z_grid,
                configuration *cfg);
int create_1d_k_grids(double **kx_grid, double **ky_grid, double **kz_grid,
                configuration *cfg);
int fill_k_grid(double *grid, int n_local, int n_global, int local_start,
                double dk);

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
    cfg.Nx_local = (int) Nx_local;
    cfg.x_start_local = (int) x_start_local;
    fftw_complex *psi_local;
    fftw_plan p_fwd, p_bwd;

    psi_local = fftw_alloc_complex(alloc_local);
    create_plans(&cfg, &p_fwd, &p_bwd, psi_local);

    printf("task %d/%d. x_start_local:%d Nx_local:%d\n", rank, size,
            x_start_local, cfg.Nx_local);

    // Create one dimensional grids for all three dimensions
    double *x_grid, *y_grid, *z_grid;
    double *kx_grid, *ky_grid, *kz_grid;
    create_1d_grids(&x_grid, &y_grid, &z_grid, &cfg);
    create_1d_k_grids(&kx_grid, &ky_grid, &kz_grid, &cfg);

    // clean up

    free(x_grid);
    free(y_grid);
    free(z_grid);
    free(kx_grid);
    free(ky_grid);
    free(kz_grid);
    fftw_destroy_plan(p_fwd);
    fftw_destroy_plan(p_bwd);
    fftw_free(psi_local);
    MPI_Finalize();

    return 0;
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


// Fill x/y/z_grids with values.
// x_grid is filled with x values using the local x index. This is different
// for every MPI process.
// The range of the values is from -(N-1)*dx/2 to (N-1)*dx/2. This ensures a
// symmetry along the axis for even values of N, which we have.
int create_1d_grids(double **x_grid, double **y_grid, double **z_grid,
                configuration *cfg) {

    *x_grid = (double*) malloc(cfg->Nx_local * sizeof(double));
    *y_grid = (double*) malloc(cfg->Ny * sizeof(double));
    *z_grid = (double*) malloc(cfg->Nz * sizeof(double));

    double x_center = (cfg->Nx - 1)/2.0 * cfg->dx;
    double y_center = (cfg->Ny - 1)/2.0 * cfg->dy;
    double z_center = (cfg->Nz - 1)/2.0 * cfg->dz;

    int i;
    for(i = 0; i < cfg->Nx_local; ++i)
        (*x_grid)[i] = (cfg->x_start_local + i) * cfg->dx - x_center;
    for(i = 0; i < cfg->Ny; ++i)
        (*y_grid)[i] = i * cfg->dy - y_center;
    for(i = 0; i < cfg->Nz; ++i)
        (*z_grid)[i] = i * cfg->dz - z_center;
    return 0;
}

int create_1d_k_grids(double **kx_grid, double **ky_grid, double **kz_grid,
                configuration *cfg) {

    *kx_grid = (double*) malloc(cfg->Nx_local * sizeof(double));
    *ky_grid = (double*) malloc(cfg->Ny * sizeof(double));
    *kz_grid = (double*) malloc(cfg->Nz * sizeof(double));

    fill_k_grid(*kx_grid, cfg->Nx_local, cfg->Nx, cfg->x_start_local,
                cfg->dkx);
    fill_k_grid(*ky_grid, cfg->Ny, cfg->Ny, 0, cfg->dky);
    fill_k_grid(*kz_grid, cfg->Ny, cfg->Ny, 0, cfg->dkz);
}

int fill_k_grid(double *grid, int n_local, int n_global, int local_start,
                double dk) {
    int i_local, i_global;
    for(i_local = 0; i_local < n_local; ++i_local) {
        i_global = i_local + local_start;
        if (i_global < n_global / 2)
            grid[i_local] = i_global * dk;
        else
            grid[i_local] = (i_global - n_global) * dk;
    }
}
