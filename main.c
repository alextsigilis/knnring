#include <stdio.h>
#include <stdlib.h>
#include "knnring.h"
#include "utils.h"

int main (int argc, char *argv[]) {

	// Declaring Variables
	FILE *in, *out;	
	int n, m, d, 
			k = atoi(argv[1]);
	double *X, *Y;
	knnresult res;

	
	// Read Input
	in = fopen("data.in", "r");
	fscanf(in, "%d %d %d", &n, &m, &d);
	X = malloc(n*d*sizeof(double));
	Y = malloc(m*d*sizeof(double));
	for(int i = 0; i < n; i++){
		double tmp;
		for(int j = 0; j < d; j++){
			fscanf(in, "%lf", &tmp);
			X[i*d+j] = tmp;
		}
	}	
	for(int i = 0; i < m; i++){ 
		double tmp;
		for(int j = 0; j < d; j++){
			fscanf(in, "%lf", &tmp);
			Y[i*d+j] = tmp;
		}
	}
	fclose(in);
	
	// Do the computation
	res = kNN(X,Y,n,m,d,k);

	// Print the Output
	out = fopen("data.out", "w");
	fprintf(out, "%d %d\n", res.m, res.k);
	for(int i = 0; i < res.m; i++){
		for(int j = 0; j < res.k; j++){
			fprintf(out, "%d ", res.nidx[i*res.k+j]);
		} fprintf(out, "\n");
	}
	for(int i = 0; i < res.m; i++){
		for(int j = 0; j < res.k; j++){
			fprintf(out, "%lf ", res.ndist[i*res.k+j]);
		} fprintf(out, "\n");
	}
	fclose(out);

	return 0;
}
