CC=mpicc
CFLAGS=-O2
LIBS=-lfftw3 -lfftw3_mpi

all: gpe_mpi.c ini.c
	$(CC) $(CFLAGS) $(LIBS) ini.c gpe_mpi.c -o gpe_mpi
