#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>

#include "common.h"
#include "configuration.h"

int read_config(configuration *cfg, int argc, char **argv) {
    char *input_filename = "parms.ini";
    // read inputs from file
    if(rank == MASTER_RANK) {
        int success = FALSE;
        int c;
        FILE *fp;
        success = TRUE;
        while ((c = getopt(argc, argv, "f:")) != EOF) {
            switch(c) {
            case 'f':
                input_filename = optarg;
                success = TRUE;
                break;
            } // end of switch(c)
        } // end of while

        if (ini_parse(input_filename, handler, cfg) < 0) {
                fprintf(stderr, "Cannot load parms file: %s'\n", input_filename);
                success = FALSE;
        }
        if(check_config(cfg) != 0)
            success = FALSE;
        process_config(cfg);
        if(!success)
            MPI_Abort(MPI_COMM_WORLD, 1);
        print_configuration(cfg);
    }
    // send configuration to all processors
    MPI_Bcast(cfg, sizeof(configuration), MPI_CHAR, MASTER_RANK,
              MPI_COMM_WORLD);
}


// Returns 1 if something wrong with config file, prints error to stderr
int check_config(configuration *cfg) {
    // find if Nx, Ny, Nz are powers of 2
    if(!POW2(cfg->Nx)) {
        fprintf(stderr, "Nx:%d is not a power of 2\n", cfg->Nx);
        return 1;
    }
    if(!POW2(cfg->Ny)) {
        fprintf(stderr, "Ny:%d is not a power of 2\n", cfg->Ny);
        return 1;
    }
    if(!POW2(cfg->Nz)) {
        fprintf(stderr, "Nz:%d is not a power of 2\n", cfg->Nz);
        return 1;
    }
    return 0;

}

// Calculate quantities required for the simulation based on inputs in
// configuration
int process_config(configuration *cfg) {

    cfg->dkx = 2.0*PI/(cfg->dx * cfg->Nx);
    cfg->dky = 2.0*PI/(cfg->dy * cfg->Ny);
    cfg->dkz = 2.0*PI/(cfg->dz * cfg->Nz);

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
    else if (MATCH("sim", "dy"))
        pconfig->dy = atof(value);
    else if (MATCH("sim", "dz"))
        pconfig->dz = atof(value);
    else if (MATCH("sim", "fx"))
        pconfig->fx = atof(value);
    else if (MATCH("sim", "fy"))
        pconfig->fy = atof(value);
    else if (MATCH("sim", "fz"))
        pconfig->fz = atof(value);
    else if (MATCH("sim", "time_scale"))
        pconfig->time_scale = atof(value);
    else if (MATCH("sim", "length_scale"))
        pconfig->length_scale = atof(value);
    else
        return 0;  /* unknown section/name, error */
    return 1;
}

int print_int(const char *variable_name, int variable) {
    printf("%s:\t%d\n", variable_name, variable);
}

int print_double(const char *variable_name, double variable) {
    printf("%s:\t%f\n", variable_name, variable);
}

int print_configuration(configuration *cfg) {
    print_int("Nx", cfg->Nx);
    print_int("Ny", cfg->Ny);
    print_int("Nz", cfg->Nz);
    printf("\n");

    print_int("Nt", cfg->Nt);
    print_int("Nt_store", cfg->Nt_store);
    print_double("dt", cfg->dt);
    printf("\n");

    print_double("dx", cfg->dx);
    print_double("dy", cfg->dy);
    print_double("dz", cfg->dz);
    printf("\n");

    print_double("dkx", cfg->dx);
    print_double("dky", cfg->dy);
    print_double("dkz", cfg->dz);
    printf("\n");

    print_double("time_scale", cfg->time_scale);
    print_double("length_scale", cfg->length_scale);
    print_double("energy_scale", cfg->energy_scale);
    printf("\n");

    print_double("fx", cfg->fx);
    print_double("fy", cfg->fy);
    print_double("fz", cfg->fz);
    printf("\n");

    print_double("K_mult", cfg->K_mult);
    print_double("U_mult", cfg->U_mult);
    print_double("I_mult", cfg->I_mult);
    printf("\n");
}
