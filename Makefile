CC=mpicc
CFLAGS=-O2

all: gpe_mpi.c
	$(CC) $(CFLAGS) gpe_mpi.c -o gpe_mpi
