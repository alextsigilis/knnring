/*
 * File		: knnring_sequential.c

 * Title	: Sequential kNN Ring

 * Short	: A brute force sollution to full k nearest neighbors.

 * Long 	: -

 * Author : Αλέξανδρος Τσιγγίλης

 * Date		: 13 November 2019

 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_blas.h>	
#include <math.h>
#include "knnring.h"

#define IDX(d,i,j) 		(d*i)+j


void product (double *c, double *a, double *b, int n, int m, int d) {
	
	gsl_matrix_view A = gsl_matrix_view_array(a, n, d);
  gsl_matrix_view B = gsl_matrix_view_array(b, m, d);
  gsl_matrix_view C = gsl_matrix_view_array(c, n, m);

  /* Compute C = A B */

  gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                  -2.0, &A.matrix, &B.matrix,
                  0.0, &C.matrix);

	c = C.matrix.data;

}

void compute_distances (double *D, double *X, double *Y, int n, int m, int d) {

	
	double *x_sq = malloc(n*sizeof(double)),
				 *y_sq = malloc(m*sizeof(double));

	// Sum of squares of X
	for(int i = 0; i < n; i++)
		for(int j = 0; j < d; j++)
			x_sq[i] += pow( X[ IDX(d,i,j) ], 2);

	
	// -2 time the product
	product(D,X,Y,n,m,d);

	// Sum of squares of Y
	for(int i = 0; i < m; i++)
		for(int j = 0; j < d; j++)
			y_sq[i] += pow( Y[ IDX(d,i,j) ], 2);


	
	// Add it all up
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			D[ IDX(m,i,j) ] += x_sq[i] + y_sq[j];
		}
	}


	return;
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k) {

	double *D = malloc(n*m*sizeof(double));

	compute_distances(D,X,Y,n,m,d);

	knnresult *result = malloc(sizeof(knnresult));
	return *result;
}
