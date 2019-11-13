CC = mpicc

CFLAGS = -O3 -Wall -g

INC = -Iinc/

TYPES = sequential

SRC = knnring

LIBS = -lm -lgsl -lgslcblas

MAIN = main

all: $(addprefix $(MAIN)_, $(TYPES))

$(MAIN)_%: $(MAIN).c lib/$(SRC)_%.a
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LIBS)
	rm -rf *.dSYM

lib: $(addsuffix .a, $(addprefix lib/$(SRC)_, $(TYPES)))


lib/%.a: lib/%.o
	ar rcs $@ $<

lib/%.o: src/%.c
	$(CC) $(CFLAGS) $(INC) -o $@ -c $<

clean:
	rm -rf *.dSYM lib/*.a *~ $(addprefix $(MAIN)_, $(TYPES))
