CC=mpicc
CFLAGS=-O2
LIBS=-lfftw3 -lfftw3_mpi

all: gpe3d_mpi

gpe3d_mpi: gpe3d_mpi.c ini.c configuration.c
	$(CC) $(CFLAGS) $(LIBS) ini.c gpe3d_mpi.c configuration.c -o gpe3d_mpi

clean:
	rm gpe3d_mpi
