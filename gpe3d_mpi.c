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


/* struct declarations */
typedef struct expectation_values {
    double x_ex;
    double y_ex;
    double z_ex;
    double x2_ex;
    double y2_ex;
    double z2_ex;
} expectation_values;

/* end of struct declarations */

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

double norm_squared(fftw_complex *vec, int length);

double norm_4(fftw_complex *vec, int length);

double potential_energy(configuration *cfg, double x, double y, double z);

void init_psi(configuration *cfg, double *x_grid, double *y_grid,
             double * z_grid, fftw_complex *psi_local);

void x_unitary(configuration *cfg, double *x_grid, double *y_grid,
               double *z_grid, fftw_complex *psi_local,
               fftw_complex imag_pre_factor);

void k_unitary(configuration *cfg, double *kx_grid, double *ky_grid,
               double *kz_grid, fftw_complex *psi_local,
               fftw_complex imag_pre_factor);

expectation_values calculate_expectation_values(configuration *cfg,
                                                double *x_grid, double *y_grid,
                                                double *z_grid,
                                                fftw_complex *psi_local);
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

    // Initialize the wavefunction to the Thomas Fermi wavefunction
    init_psi(&cfg, x_grid, y_grid, z_grid, psi_local);

    fftw_complex imag_pre_factor;
    if(cfg.imag_time)
        imag_pre_factor = 1;
    else
        imag_pre_factor = I;

    int Nt_skip = cfg.Nt / (cfg.Nt_store - 1);


    /************************************/
    /* MAIN SIMULATION LOOP STARTS HERE */
    /************************************/
    int j = 1, i = 0;
    for(i = 0; i < cfg.Nt; i++) {

        // Do X-unitary
        x_unitary(&cfg, x_grid, y_grid, z_grid, psi_local, imag_pre_factor);

        MPI_Barrier(MPI_COMM_WORLD);
        // Fourier transform
        fftw_execute(p_fwd);

        k_unitary(&cfg, kx_grid, ky_grid, kz_grid, psi_local, imag_pre_factor);

        MPI_Barrier(MPI_COMM_WORLD);

        // Inverse fourier transform
        fftw_execute(p_bwd);

        MPI_Barrier(MPI_COMM_WORLD);

        // Do X-unitary
        x_unitary(&cfg, x_grid, y_grid, z_grid, psi_local, imag_pre_factor);

        MPI_Barrier(MPI_COMM_WORLD);
        if(cfg.imag_time) {
            // normalize the wavefunction
            double p2 = norm_squared(psi_local, alloc_local);  // get norm
            p2 *= cfg.dx * cfg.dy * cfg.dz;
            double p2_inv_sqrt = 1.0/sqrt(p2);
            // printf("p2: %f, 1/p2 %f\n", p2, p2_inv_sqrt);
            int i1;
            for(i1=0; i1 < alloc_local; i1++)
                psi_local[i1] *= p2_inv_sqrt;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if( (i + 1) % Nt_skip == 0) {
            double n4 = norm_4(psi_local, alloc_local);
            expectation_values ex_vals;
            ex_vals = calculate_expectation_values(&cfg, x_grid, y_grid,
                                                   z_grid, psi_local);
            if(rank == MASTER_RANK) {
                double mfe = n4 * cfg.kappa * cfg.dx * cfg.dy * cfg.dz;
                double pot_en = ex_vals.x2_ex +
                                ex_vals.y2_ex * cfg.gamma_y * cfg.gamma_y +
                                ex_vals.z2_ex * cfg.gamma_z * cfg.gamma_z;
                pot_en *= 0.5;
                double mu_ex = mfe + pot_en;

                printf("\n*****************************\n");
                printf("Time step %d, Store step %d\n", i, j);
                print_double("MFE", mfe);
                print_double("<X>", ex_vals.x_ex);
                print_double("<Y>", ex_vals.y_ex);
                print_double("<Z>", ex_vals.z_ex);
                print_double("<X2>", ex_vals.x2_ex);
                print_double("<Y2>", ex_vals.y2_ex);
                print_double("<Z2>", ex_vals.z2_ex);
                print_double("<U>", pot_en);
                print_double("<mu>", mu_ex);
            }
            j++;
        }
    }

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


double norm_squared(fftw_complex *vec, int length) {
    // Returns the square of the complex vector norm
    double sum_local, sum_global;
    int j;
    sum_local = 0.0;
    for(j = 0; j < length; j++)
        sum_local += vec[j] * conj(vec[j]);
    MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return sum_global;

}


double norm_4(fftw_complex *vec, int length) {
    // find local sum
    double sum_local, sum_global;
    int j;
    sum_local = 0.0;
    for(j = 0; j < length; j++) {
        double n2 = vec[j] * conj(vec[j]);
        sum_local += n2 * n2;
    }
    MPI_Reduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, MASTER_RANK,
                  MPI_COMM_WORLD);
    return sum_global;
}


double potential_energy(configuration *cfg, double x, double y, double z) {
    // Returns the potential energy at the point (x, y, z)
    return 0.5 * (x * x + y * y * cfg->gamma_y * cfg->gamma_y
                  + z * z * cfg->gamma_z * cfg->gamma_z);
}

void init_psi(configuration *cfg, double *x_grid, double *y_grid,
             double * z_grid, fftw_complex *psi_local) {

    int current_index = 0;
    double x_c, y_c, z_c;
    int ix, iy, iz;
    for(ix = 0; ix < cfg->Nx_local; ix++) {
        x_c = x_grid[ix];
        for(iy = 0; iy < cfg->Ny; iy++) {
            y_c = y_grid[iy];
            for(iz = 0; iz < cfg->Nz; iz++) {
                z_c = z_grid[iz];
                double pot_en = potential_energy(cfg, x_c, y_c, z_c);
                double psi2_val = (cfg->mu_theory - pot_en) / cfg->kappa;
                if(psi2_val > 0.0)
                    psi_local[current_index] = sqrt(psi2_val);
                else
                    psi_local[current_index] = 0.0;
                current_index++;
            }
        }
    }
}

void x_unitary(configuration *cfg, double *x_grid, double *y_grid,
               double *z_grid, fftw_complex *psi_local,
               fftw_complex imag_pre_factor) {
    int ix, iy, iz;
    double x_c, y_c, z_c;
    int current_index = 0;
    for(ix = 0; ix < cfg->Nx_local; ix++) {
        x_c = x_grid[ix];
        for(iy = 0; iy < cfg->Ny; iy++) {
            y_c = y_grid[iy];
            for(iz = 0; iz < cfg->Nz; iz++) {
                z_c = z_grid[iz];
                double pot_en = potential_energy(cfg, x_c, y_c, z_c);
                fftw_complex psi_c = psi_local[current_index];
                double psi2_c = psi_c * conj(psi_c);
                double phase = -(pot_en * cfg->dt * 0.5 +
                               cfg->kappa * psi2_c * cfg->dt * 0.5);
                psi_local[current_index] *= cexp(imag_pre_factor * phase);
                current_index++;
            }
        }
    }
}

void k_unitary(configuration *cfg, double *kx_grid, double *ky_grid,
               double *kz_grid, fftw_complex *psi_local,
               fftw_complex imag_pre_factor) {
    int current_index = 0;
    int ix, iy, iz;
    double kx_c, ky_c, kz_c;
    double k_normalization = 1.0/(double)(cfg->Nx * cfg->Ny * cfg->Nz);
    for(ix = 0; ix < cfg->Nx_local; ix++) {
        kx_c = kx_grid[ix];
        for(iy = 0; iy < cfg->Ny; iy++) {
            ky_c = ky_grid[iy];
            for(iz = 0; iz < cfg->Nz; iz++) {
                kz_c = kz_grid[iz];
                double k_squared = kx_c * kx_c + ky_c * ky_c + kz_c * kz_c;
                double phase = -k_squared * cfg->dt * 0.5;
                psi_local[current_index] *= cexp(imag_pre_factor * phase) *
                                            k_normalization;
                current_index++;
            }
        }
    }
}

expectation_values calculate_expectation_values(configuration *cfg,
                                                double *x_grid, double *y_grid,
                                                double *z_grid,
                                                fftw_complex *psi_local) {
    // Returns the expectation values on MASTER_RANK. Garbage for other
    // processes
    expectation_values ex_local;
    expectation_values ex_global;
    ex_local.x_ex = ex_local.y_ex = ex_local.z_ex = 0.0;
    ex_local.x2_ex = ex_local.y2_ex = ex_local.z2_ex = 0.0;

    int ix, iy, iz;
    double x_c, y_c, z_c;
    int current_index = 0;
    for(ix = 0; ix < cfg->Nx_local; ix++) {
        x_c = x_grid[ix];
        for(iy = 0; iy < cfg->Ny; iy++) {
            y_c = y_grid[iy];
            for(iz = 0; iz < cfg->Nz; iz++) {
                z_c = z_grid[iz];
                fftw_complex psi_val = psi_local[current_index];
                double p2 = psi_val * conj(psi_val);
                ex_local.x_ex += p2 * x_c;
                ex_local.y_ex+= p2 * y_c;
                ex_local.z_ex+= p2 * z_c;

                ex_local.x2_ex += p2 * x_c * x_c;
                ex_local.y2_ex += p2 * y_c * y_c;
                ex_local.z2_ex += p2 * z_c * z_c;

                current_index++;
            }
        }
    }
    MPI_Reduce(&ex_local, &ex_global, 6, MPI_DOUBLE, MPI_SUM, MASTER_RANK,
               MPI_COMM_WORLD);
    double dV = cfg->dx * cfg->dy * cfg->dz;

    ex_global.x_ex *= dV;
    ex_global.y_ex *= dV;
    ex_global.z_ex *= dV;

    ex_global.x2_ex *= dV;
    ex_global.y2_ex *= dV;
    ex_global.z2_ex *= dV;

    return ex_global;
}
