####################################################
## 				The Directory for the  		    					##
##  			BLAS (openblas) Library		    					##
## 			Set the enviroment viairable: 						##
##   																							##
## 	$ export OPENBLAS_ROOT=/path/to/open/blas 		##
##																								##
####################################################

BLAS=${OPENBLAS_ROOT}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# The Shell you're using
SHELL := /bin/bash

# The C compiler
CC = mpicc

# Flags for the gcc
CFLAGS = -O3 -Wall -g

# Include paths for header files
INC = -Iinc/ -I$(BLAS)/include/

# Paths for libriries to link
LDFLAGS = -L$(BLAS)/lib/

# Libraries to load
LIBS = -lm -lopenblas

# -----=-------=--------=-----=-------=-----=

TYPES = mpi sequential

SRC = knnring

MAIN = main

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
all: $(addprefix $(MAIN)_, $(TYPES))

$(MAIN)_%: $(MAIN).c lib/$(SRC)_%.a
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS) $(LIBS)
	rm -rf *.dSYM

lib: $(addsuffix .a, $(addprefix lib/$(SRC)_, $(TYPES)))


lib/%.a: lib/%.o
	ar rcs $@ $<

lib/%.o: src/%.c
	echo $(BLAS)
	$(CC) $(CFLAGS) $(INC) -o $@ -c $<

clean:
	rm -rf *.dSYM lib/*.a *~ $(addprefix $(MAIN)_, $(TYPES))
