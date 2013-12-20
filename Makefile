CC=mpicc
CFLAGS=-O2
LIBS=-lfftw3 -lfftw3_mpi

all: gpe_mpi.c
	$(CC) $(CFLAGS) $(LIBS) gpe_mpi.c -o gpe_mpi
