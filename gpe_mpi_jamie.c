//
//  GPEMPI.c
//  GPE vail command will only show you modules whose names start with the
//  argument that you give it, and will alsi return modules that you cannot load
//  due to conflicts with already loaded modules.
//
//  A little SciNet utility called modulefind (one word) can do that. It will
//  list all installed modules which contain the arguments, and will determine
//  whether those modules have been loaded, could be loaded, cannot because of
//  conflicts with already loaded modules, or have unresolved dependencies (i.e.
//  for which other modules need to be loaded first). This is especially useful
//  in cases like the "boost" libraries, whose module names are
//  cxxlibraries/boost/1.47.0-gcc and cxxlibraries/boost/1.47.0-gcc, for the gcc
//  and intel compiler, respectively. modulefind boost will find those, whereas
//  module avail boost will not.
//
//  Note that just 'modulefind' will list all top-level modules. Simulation
//
//  Created by Donald Woodbury on 2013-03-10.
//  Copyright (c) 2013 Donald Woodbury. All rights reserved.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>

const int n_skip = 1;  //Number of time steps to skip between snapshots of the array being saved to files
const fftw_complex imag_time = I;   //Defines whether the program runs in real time

//Spatial array sizes
const ptrdiff_t x_size = 128;
const ptrdiff_t y_size = 128;
const ptrdiff_t z_size = 128;

//GPE Constants
const double pi = 3.14159265;
const double epsilon = 1;
const double epsilon1 = 0.25;
const double Kd = 0;
const double gammay = 1;
const double gammaz = 1;

//Time and Space discretization intervals and the total time for which the simulation will run
const double dt = 0.01;
const double dx = 0.0625;
const double endt = 5;

//Constants for decreasing the resolution and dimension of the array for exporting purposes.

const int x_red_factor = 2;
const int y_red_factor = 2;
const int z_red_factor = 128;

void def_xyz(double* xyz, int dim, ptrdiff_t local_x_size , ptrdiff_t local_x_start){
    // Defines the x, y, or z arrays depending on the value of dim

    int i;
    if(dim == 1){
        for(i=0; i < local_x_size; i++){
            float size = (float) x_size;
            xyz[i] = (i + local_x_start-size/2)*dx;
        }
    }
    if(dim == 2){
        for(i=0; i < y_size; i++){
            float size = (float) y_size;
            xyz[i] = (i-size/2)*dx;
        }
    }
    if(dim == 3){
        for(i=0; i < z_size; i++){
            float size = (float) z_size;
            xyz[i] = (i-size/2)*dx;
        }
    }
}

void def_K(double* K, int dim, ptrdiff_t local_x_size , ptrdiff_t local_x_start){
    //Defines the x, y, or z frequency array depending on the value of dim
    
    float size;
    int start;
    int stop;
    
    
    if(dim == 1){
        size = (float) x_size;
        start = (int) local_x_start;
        stop = ((int) local_x_start) + ((int) local_x_size);
    }
    if(dim == 2){
        size = (float) y_size;
        start = 0;
        stop = y_size;
    }
    if(dim == 3){
        size = (float) z_size;
        start = 0;
        stop = z_size;
    }
    
    int i;
    for(i= start; i < stop; i++){
        if(i<(size/2)){
            K[i-start] = 2*pi*i/(dx*size);
        }
        else{
            K[i-start] = 2*pi*(i-size)/(dx*size);
        }
    }
}

void def_Ux(double* Ux, double* x, double* y, double* z, ptrdiff_t local_x_size, ptrdiff_t local_x_start){
    //Defines the 3D harmonic potential array
    
    int i, j, k;
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                Ux[k + z_size * (j + y_size * i)] = 0.5*(pow(x[i], 2) + pow(gammay*y[j],2) + pow(gammaz*z[k], 2));
            }
        }
    }
}

void psi_init(fftw_complex* psi, double* x, double* y, double* z, ptrdiff_t local_x_size){
    //Defines the 3D ground state solution to the quantum harmonic oscilator
    
    int i, j, k;
    double prefactor = pow(gammay*gammaz, 0.25)/pow(pi*epsilon1, 0.75);
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                psi[k + z_size * (j + y_size * i)] = 1;
                //prefactor * exp(-(pow(x[i], 2) + gammay*pow(y[j],2) + gammaz*pow(z[k], 2))/(2*epsilon1));
            }
        }
    }
}

void def_SHO_psi(fftw_complex* SHO_psi, double* x, double* y, double* z, ptrdiff_t local_x_size){
    //Defines the 3D ground state solution to the quantum harmonic oscilator
    
    int i, j, k;
    double prefactor = pow(gammay*gammaz, 0.25)/pow(pi*epsilon, 0.75);
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                SHO_psi[k + z_size * (j + y_size * i)] =
                prefactor * exp(-(pow(x[i], 2) + gammay*pow(y[j],2) + gammaz*pow(z[k], 2))/(2*epsilon));
            }
        }
    }
}

void def_TF_psi(fftw_complex* TF_psi, double* Ux, long local_x_size){
    //Defines the 3D Thomas-Fermi solution to the GPE
    
    long i;
    for(i = 0; i < local_x_size*y_size*z_size; i++){
        
        double mu;
        mu = 0.5*pow(15*Kd*gammay*gammaz/(4*pi), 2.0/5.0);
        
        if(Ux[i] < mu){
            TF_psi[i] = sqrt((mu-Ux[i])/Kd);
        }
        else{
            TF_psi[i] = 0;
        }
    }
}

void x_unitary(fftw_complex* psi, double* Ux, ptrdiff_t local_x_size){
    //Performs the spatial unitary transformation, in place, on the array psi
    
    int i;
    for(i = 0; i < local_x_size * y_size * z_size; i++){
        psi[i] = psi[i] * cexp(imag_time*I*(Ux[i] + Kd*pow(cabs(psi[i]), 2))*dt/epsilon);
    }
}

void k_unitary(fftw_complex* psi, double* Kx, double* Ky, double* Kz, ptrdiff_t local_x_size){
    //Performs the frequency space unitary transformation, in place, on the array psi. The normalization
    //factor "prefactor" is included to renormalize the array after the fourier transform, since in the
    //fftw module ifft(fft(A)) = x_size * y_size * z_size * A
    
    int i, j, k;
    
    double prefactor;
    prefactor = 1.0/((double) (x_size * y_size * z_size));
    
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                fftw_complex PSI;
                PSI = psi[k + z_size * (j + y_size * i)];
                psi[k + z_size * (j + y_size * i)] =
                prefactor * PSI * cexp(imag_time * I * (pow(Kx[i], 2) + pow(Ky[j],2) + pow(Kz[k], 2))*epsilon*dt/2);
            }
        }
    }
}

void normalize_psi(fftw_complex* psi, int rank, int size, ptrdiff_t local_x_size){
    //Normalizes the modulo square of the array psi. Used when performing the imaginary time-stepping.
    
    long j;
    double sum_local, sum;
    for(j = 0; j < local_x_size * y_size * z_size; j++){
        sum_local += pow(cabs(psi[j]), 2)*pow(dx, 3);
    }
    
    if(rank == 0){
        double sum_buff;
        
        sum = sum_local;
        
        int i;
        for(i=1; i<size; i++){
            MPI_Recv(&sum_buff, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += sum_buff;
        }
    }
    else{
        MPI_Ssend(&sum_local, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for(j = 0; j < local_x_size * y_size * z_size; j++){
        psi[j] = psi[j]/sqrt(sum);
    }
    //printf("Rank %d, Sum: %f\n", rank, sum);
}

void shrink_psi(fftw_complex* psi_in, double* psi_out, long local_x_size){
    //Reduces the resolution of psi by averaging over the modulo squared the each n varriables defined by the reduction factors
    //(e.g. x_red_factor).
    
    long target_x_size, target_y_size, target_z_size;
    
    target_x_size = local_x_size / x_red_factor;
    target_y_size = y_size / y_red_factor;
    target_z_size = z_size / z_red_factor;
    
    long l;
    for(l=0; l < target_x_size * target_y_size * target_z_size; l++){
        psi_out[l] = 0;
    }
    
    long i, j, k;
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                long index;
                index = (k/z_red_factor) + target_z_size * ((j/y_red_factor) + target_y_size * (i/x_red_factor));
                psi_out[index]+= pow(cabs(psi_in[k + z_size * (j + y_size * i)]), 2);
            }
        }
    }
}

void write_to_file(fftw_complex* psi, long local_x_size, MPI_File file){
    //writes the locally stored section of the array (at a given time step) to a file.
    
    //Taking |psi|^2 and reducing the Resolution and dimension
    
    long target_x_size, target_y_size, target_z_size;
    
    target_x_size = local_x_size / x_red_factor;
    target_y_size = y_size / y_red_factor;
    target_z_size = z_size / z_red_factor;
    
    double *psi_out;
    psi_out = (double*) malloc(target_x_size * target_y_size * target_z_size * sizeof(double));
    
    shrink_psi(psi, psi_out, local_x_size);
    
    //Saving to file
    
    MPI_File_write_ordered(file, psi_out, (int) (target_x_size * target_y_size * target_z_size), MPI_DOUBLE, MPI_STATUS_IGNORE);
    
    //Cleaning up
    
    free(psi_out);
}

double find_error(fftw_complex* psi, fftw_complex* psi0, long local_x_size, int rank, int size){
    //Calculates the error in the absolute value squared of two arrays psi and psi0
    
    double error, error_local;
    error_local = 0;
    
    long i;
    for(i = 0; i < local_x_size*y_size*z_size; i++){
        error_local += cabs(pow(cabs(psi0[i]), 2)-pow(cabs(psi[i]), 2))*pow(dx, 3);
    }
    
    //Process 0 collects all of the local sums and then distributes the total to each process
    if(rank == 0){
        
        double error_buff;
        error = error_local;
        
        int i;
        for(i=1; i<size; i++){
            MPI_Recv(&error_buff, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            error += error_buff;
        }
    }
    else{
        MPI_Ssend(&error_local, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return error;
}

double pos_expectation(fftw_complex* psi, double* xyz, long local_x_size, int dim, int rank, int size){
    //Calculates the expectation value of the x, y, or z position of psi depending on the value of dim given
    
    double sum_local, sum;
    
    //Calculating the local contributions to the integral
    long i, j, k;
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                if(dim==1){
                    sum_local += xyz[i] * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
                if(dim==2){
                    sum_local += xyz[j] * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
                if(dim==3){
                    sum_local += xyz[k] * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
            }
        }
    }
    
    //Collecting the local sums onto the first processor
    if(rank == 0){
        double sum_buff;
        
        sum = sum_local;
        
        int i;
        for(i=1; i < size; i++){
            MPI_Recv(&sum_buff, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += sum_buff;
        }
    }
    else{
        MPI_Ssend(&sum_local, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    //Distributing the result to all processors
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return sum;
}

double condensate_size(fftw_complex* psi, double* x, double* y, double* z, long local_x_size, int dim, int rank, int size){
    //Calculating the standard deviation of the condensate in either the x, y, or z direction
    
    double mean;
    
    if(dim ==1){
        mean = pos_expectation(psi, x, local_x_size, 1, rank, size);
    }
    if(dim ==2){
        mean = pos_expectation(psi, y, local_x_size, 2, rank, size);
    }
    if(dim ==3){
        mean = pos_expectation(psi, z, local_x_size, 3, rank, size);
    }
    
    double sum_local, sum;
    
    //Calculating the local contributions to the integral
    long i, j, k;
    for(i = 0; i < local_x_size; i++){
        for(j = 0; j < y_size; j++){
            for(k = 0; k < z_size; k++){
                if(dim==1){
                    sum_local += pow(x[i]-mean, 2) * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
                if(dim==2){
                    sum_local += pow(y[j]-mean, 2) * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
                if(dim==3){
                    sum_local += pow(z[k]-mean, 2) * pow(cabs(psi[k + z_size * (j + y_size * i)]), 2)*pow(dx, 3);
                }
            }
        }
    }
    
    //Collecting the local sums onto the first processor
    if(rank == 0){
        double sum_buff;
        
        sum = sum_local;
        
        int i;
        for(i=1; i < size; i++){
            MPI_Recv(&sum_buff, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += sum_buff;
        }
    }
    else{
        MPI_Ssend(&sum_local, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    //Distributing the result to all processors
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return sqrt(sum);
}

int main(int argc, char *argv[]){
    
    //Initialize MPI and fftw_MPI, and get local info, size and rank
    
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rank_copy;
    rank_copy = rank;
    
    ptrdiff_t alloc_local, local_x_size, local_x_start;
    alloc_local = fftw_mpi_local_size_3d(x_size, y_size, z_size, MPI_COMM_WORLD, &local_x_size, &local_x_start);
    
    //Allocating memory and defining spatial and frequency, potential arrays
    
    double *x, *y, *z;
    
    x = (double*) malloc(local_x_size * sizeof(double));
    y = (double*) malloc(y_size * sizeof(double));
    z = (double*) malloc(z_size * sizeof(double));
    
    def_xyz(x, 1, local_x_size, local_x_start);
    def_xyz(y, 2, local_x_size, local_x_start);
    def_xyz(z, 3, local_x_size, local_x_start);
    
    double *Kx, *Ky, *Kz;
    
    Kx = (double*) malloc(local_x_size * sizeof(double));
    Ky = (double*) malloc(y_size * sizeof(double));
    Kz = (double*) malloc(z_size * sizeof(double));
    
    def_K(Kx, 1, local_x_size, local_x_start);
    def_K(Ky, 2, local_x_size, local_x_start);
    def_K(Kz, 3, local_x_size, local_x_start);
    
    double *Ux;
    
    Ux = (double*) malloc(local_x_size * y_size * z_size * sizeof(double));
    
    def_Ux(Ux, x, y, z, local_x_size, local_x_start);
    
    //Here we plan the fourier transform to be performed by fftw. We use the plan function to determine the amount of memory
    //to allocate psi, since it also leaves room for the computations to occur.
    
    fftw_complex *psi_local;
    fftw_plan pforward, pbackward;
    
    psi_local = fftw_alloc_complex(alloc_local);
    
    pforward = fftw_mpi_plan_dft_3d(x_size, y_size, z_size, psi_local, psi_local, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    pbackward = fftw_mpi_plan_dft_3d(x_size, y_size, z_size, psi_local, psi_local, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    
    psi_init(psi_local, x, y, z, local_x_size);
    
    //Opening output files
    
    FILE* condensate_widths_file;
    char widths_filename [50];
    sprintf(widths_filename, "/scratch/a/aephraim/dwoodbur/result/widths.txt");
    
    if(rank==0){
        condensate_widths_file = fopen(widths_filename, "w");
    }
    
    FILE* error_file;
    char error_filename [50];
    sprintf(error_filename, "/scratch/a/aephraim/dwoodbur/result/error.txt");
    
    if(rank==0){
        error_file = fopen(error_filename, "w");
    }
    
    MPI_File binary_dump;
    char filename [50];
    sprintf(filename, "/scratch/a/aephraim/dwoodbur/result/psi.bin");
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &binary_dump);
    
    //Now we iterate over time using the time step dt. MPI Barriers are included to sync all tasks before performing the fft
    
    double t;
    for(t = 0; t < (endt-dt/2); t = t + dt){
        
        //performing the Split-Step Fourier method
        
        x_unitary(psi_local, Ux, local_x_size);
        
        MPI_Barrier(MPI_COMM_WORLD);
        fftw_execute(pforward);
        k_unitary(psi_local, Kx, Ky, Kz, local_x_size);
        MPI_Barrier(MPI_COMM_WORLD);
        fftw_execute(pbackward);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(imag_time == I){
            normalize_psi(psi_local, rank_copy, size, local_x_size);
        }
        
        //Comparing the result to the expected value for psi in the imaginary time stepping case
        
        if(imag_time == I){
            double error;
            fftw_complex *TF_psi;
            fftw_complex *SHO_psi;
            
            TF_psi = (fftw_complex*) malloc(local_x_size * y_size * z_size * sizeof(fftw_complex));
            SHO_psi = (fftw_complex*) malloc(local_x_size * y_size * z_size * sizeof(fftw_complex));
            
            def_TF_psi(TF_psi, Ux, local_x_size);
            def_SHO_psi(SHO_psi, x, y, z, local_x_size);
            
            error = find_error(psi_local, SHO_psi, local_x_size, rank, size);
            if(rank == 0){
                printf("t: %f, Error: %f\n", t, error);
                fprintf(error_file, "%f\t%f\n", t, error);
            }
            free(TF_psi);
            free(SHO_psi);
        }
        
        //Now we calculate the condensate size in the real time stepping case
        
        if(imag_time == 1){
        
            double sigmax, sigmay, sigmaz;
            
            sigmax = condensate_size(psi_local, x, y, z, local_x_size, 1, rank, size);
            sigmay = condensate_size(psi_local, x, y, z, local_x_size, 2, rank, size);
            sigmaz = condensate_size(psi_local, x, y, z, local_x_size, 3, rank, size);
            
            if(rank==0){
                printf("t: %f, X size: %f, Y size: %f, Z size: %f\n", t, sigmax, sigmay, sigmaz);
                fprintf(condensate_widths_file, "%f\t%f\t%f\t%f\n", t, sigmax, sigmay, sigmaz);
            }
        }
        
        //Here we save a snapshot of psi to a binary file
        
        if(((int) round(t/dt))%n_skip == 0){
            write_to_file(psi_local, local_x_size, binary_dump);
        }
    }
    
    //Cleaning up
    
    fftw_destroy_plan(pforward);
    fftw_destroy_plan(pbackward);
    
    fftw_free(psi_local);
    free(Ux);
    
    free(x);
    free(y);
    free(z);
    
    free(Kx);
    free(Ky);
    free(Kz);
    
    if(rank == 0){
        fclose(condensate_widths_file);
        fclose(error_file);
    }
    MPI_File_close(&binary_dump);
    
    MPI_Finalize();
}
