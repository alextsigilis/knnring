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
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <cblas.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "knnring.h"
#include "utils.h"


//! Computes the product -2 * A*B' and assigns it to C
void product(double *C, double *A, double *B, int n, int m, int d) {

  uint64_t A_rows  =   n,
      		A_cols  =   d,
      		lda     =   A_cols,
      		B_rows  =   m,
      		B_cols  =   d,
      		ldb     =   B_cols,
      		C_cols  =   B_rows,
      		ldc     =   C_cols;

	cblas_dgemm(CblasRowMajor,
							CblasNoTrans, CblasTrans, A_rows, B_rows, A_cols,
							-2.0, A, lda, B, ldb,
							1.0, C, ldc
						 );
	return;

}

//! Computes the distance of every point in X to every point in Y
void compute_distances (double *D, double *X, double *Y, int n, int m, int d) {


	// Setting the leading dimetions for each matrix
 	uint64_t ldx = d, ldy = d, ldD = n;
	// Declaring the vectors for the squares of X and Y,
	// x_sq = sum(X.^2,2), y_sq = sum(Y.^2,2)
	double x_sq[n], y_sq[m];


	// Initializing to zero
  cilk_for(uint64_t i = 0; i < n*m; i++){
		D[i] = 0;
		if (i < m) y_sq[i] = 0;
		if (i < n) x_sq[i] = 0;
	}

	// x_sq = sum(X.^2,2)
  cilk_for(uint64_t i = 0; i < n; i++)
		for(uint64_t j = 0; j < d; j++) x_sq[i] += pow( X[i*ldx+j], 2);

	// y_sq = sum(Y.^2,2)
  cilk_for(uint64_t i = 0; i < m; i++)
		for(int j = 0; j < d; j++) y_sq[i] += pow( Y[i*ldy+j], 2);

	//D += x_sq' + y_sq
  cilk_for(uint64_t i = 0; i < m; i++)
    cilk_for(uint64_t j = 0; j < n; j++) {
			D[i*ldD+j] = x_sq[j] + y_sq[i];
		}

	// D += -2*Y*X'
  product(D,Y,X,m,n,d);

	// Taking the square root
  cilk_for(uint64_t i = 0; i < n*m; i++) D[i] = sqrt(D[i]);

	return;
}


//! Partitions the elemets of X
int qPartition(double *X, int *idx, int n) {

  int i = -1;
  int pivot = n-1;
  for(int j=0; j < n-1; j++){
    if(X[j] <= X[pivot]){
      double tmp;
      tmp = X[++i];
      X[i] = X[j];
      X[j] = tmp;

			tmp = (double) idx[i];
			idx[i] = idx[j];
			idx[j] = (int) tmp;

    }
  }

  double tmp;
  tmp = X[++i];
  X[i] = X[pivot];
  X[pivot] = tmp;

	tmp = (double) idx[i];
	idx[i] = idx[pivot];
	idx[pivot] = (int) tmp;

  return i;


}

//! "Selects" the k-th smallest element in X,
//! (it also partitions the array in smaller and bigger elements)
double qSelect(double *X, int *idx, int n, int k) {

  if (n == 1) {
    return X[0];
  }

  int r = qPartition(X,idx, n);

  if (r < k) {
    return qSelect(X+r, idx+r, n-r, k);
  }
  else if (r == k) {
    return X[r];
  }
  else {
    return qSelect(X, idx, r, k);
  }
}


void qSort(double *X, int *idx, int n) {

	if (n <= 1) {
		return;
	}

	int r = qPartition(X,idx,n);

	qSort(X, idx, r);
	qSort(X+r, idx+r, n-r);

}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k) {

	// Allocating memory for the distance matrix (m-by-n)
	double *D 			= 	(double*)malloc( (uint64_t)n * (uint64_t)m * sizeof(double));
	
	// Allocating memory for the index array
	int*idx = malloc(n*sizeof(int)),
			ldD = n,
			ldr;

	// Initializing the result variable
	knnresult *result = malloc(sizeof(knnresult));
	result->m 		= 	m;
	result->k			=		k;
	result->nidx 	= 	malloc(result->m * result->k * sizeof(int));
	result->ndist = 	malloc(result->m * result->k * sizeof(double));
			ldr 			=		result->k; 
	

	// Compute the distances of every corpus point,
	// to every query point
	compute_distances(D,X,Y,n,m,d);


	// Find the knn's
	for(uint64_t qp = 0; qp < m; qp++) { 			// Foreach query point in Y

		// Set the indexes
		for(uint64_t i = 0; i < n; i++) idx[i] = i;

		// Find the k-smallest elements in D[qp,:]
		qSelect(D+ldD*qp, idx, n, k);

		// Set the values in the result
		for(uint64_t cp = 0; cp < k; cp++) {

			result->nidx[ qp*ldr+cp ] = idx[ cp ];
			result->ndist[  qp*ldr+cp ] = D[ qp*ldD + cp  ];
		}

		// Sort the result
		qSort(
					result->ndist +qp*ldr, 
					result->nidx+qp*ldr,
					k
				);

	}

	return *result;
}
