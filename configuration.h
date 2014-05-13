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

    // simulation specific stuff
    double gamma_y, gamma_z;
    double kappa;

    // calculated stuff
    double mu_theory;
    double rx_theory, ry_theory, rz_theory;

    int imag_time;

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
