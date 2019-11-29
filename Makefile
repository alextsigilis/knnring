CC = mpicc

CFLAGS = -O3 -Wall -g -fsanitize=address 

# The Path to the OpenBLAS library
OPENBLAS = /usr/local/Cellar/openblas/0.3.7

INC = -Iinc/ -I$(OPENBLAS)/include/

LDFLAGS = -L$(OPENBLAS)/lib/

TYPES = mpi

SRC = knnring

LIBS = -lm -lopenblas

MAIN = main

# ----------------------------------------------
all: $(addprefix $(MAIN)_, $(TYPES))

$(MAIN)_%: $(MAIN).c lib/$(SRC)_%.a
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS) $(LIBS)
	rm -rf *.dSYM

lib: $(addsuffix .a, $(addprefix lib/$(SRC)_, $(TYPES)))


lib/%.a: lib/%.o
	ar rcs $@ $<

lib/%.o: src/%.c
	$(CC) $(CFLAGS) $(INC) -o $@ -c $<

clean:
	rm -rf *.dSYM lib/*.a *~ $(addprefix $(MAIN)_, $(TYPES))
