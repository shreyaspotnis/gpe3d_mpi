#ifndef CONFIGURATION_H
#define CONFIGURATION_H

typedef struct configuration {
    int Nx;
    int Nx_local;
    int x_start_local;
    int Ny;
    int Nz;
    int Nt;
    int Nt_store;
    double dt;
    double dx, dy, dz;
    double dkx, dky, dkz;

    double time_scale;
    double length_scale;
    double energy_scale;

    // simulation specific stuff
    double fx, fy, fz; // for a harmonic oscillator potential

    double K_mult; // multiplier for the kinetic energy term
    double U_mult; // multiplier for the potential energy term
    double I_mult; // multiplier for the interaction energy term
} configuration;

int read_config(configuration *cfg, int argc, char **argv);
int check_config(configuration *cfg);
int process_config(configuration *cfg);
int print_config(configuration *cfg);

int print_int(const char *variable_name, int variable);
int print_double(const char *variable_name, double variable);

// for reading INI files
static int handler(void* user, const char* section, const char* name,
                   const char* value);

#endif
