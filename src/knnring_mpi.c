/*
 * File		: knnring_sequential.c

 * Title	: Synchronus kNN Ring

 * Short	:

 * Long 	: -

 * Author : Αλέξανδρος Τσιγγίλης

 * Date		: 29 November 2019

*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <cblas.h>
#include <float.h>
#include <time.h>
#include "knnring.h"

#define		even(n)				((n % 2) == 0)
#define 	odd(n)				(!even(n))
#define		prev(pid)			((P+pid-1)%P)
#define		next(pid)			((pid+1)%P)
#define		offset(p)			(n*prev(pid-p))

#define		knndist(i,j)	(knn.ndist[i*k+j])
#define		knnidx(i,j)		(knn.nidx[i*k+j])
#define		resdist(i,j)	(res.ndist[i*k+j])
#define		residx(i,j)		(res.nidx[i*k+j])
#define		TAG								1

//! Inserts a key to the Heap (arrays dist and idx)
/*!
	\param dist			Array that holds the values			[n-by-1]
	\param idx			Array of the initial indexes		[n-by-1]
	\param n				The length of the arrays				[scalar]
	\param v				The `key` to be inserted				[scalar]
	\param id				The initial index of the key		[scalar]
	---------------------------------------------------------
	\return 		`void`

*/
void insert(double *dist, int *idx, double v, int id, int k){

	int i = k-1;

	if( v > dist[i] ){
			return;
	}

	dist[i] = v;
	idx[i] = id;

	while( i > 0 && dist[i-1] > dist[i] ){
		double tmp_d = dist[i];
		dist[i] = dist[i-1];
		dist[i-1] = tmp_d;

		int tmp_i = idx[i];
		idx[i] = idx[i-1];
		idx[i-1] = tmp_i;

		i--;
	}
	return;
}


//! Returns the element of which k elements are less than or equal,
//!		it also partitions the arrays
/*!
	\param dist 		The array of elements								[n-by-1]
	\param idx			The indexes in the initial array		[n-by-1]
	\param n				The number of the elements					[scalar]
	\param k				The dimentions of the points				[scalar]
	-------------------------------------------------------------
	\return 				The k-th element
*/
double quickSelect (double *dist, int *idx, int n, int k) {
	int start = 0;
	int end = n;

	while (start != end) {

		//
		// PARTITION
		//
		double pivot = dist[end-1];

		int i = start-1;

		for (int j = start; j < end-1; j++) {
			if ( dist[j] <= pivot ) {
				i++;
				double tmp_d = dist[j];
				dist[j] = dist[i];
				dist[i] = tmp_d;

				int tmp_i = idx[j];
				idx[j] = idx[i];
				idx[i] = tmp_i;
			}
		}

		double tmp_d= dist[i+1];
		dist[i+1] = dist[end-1];
		dist[end-1] = tmp_d;

		int tmp_i = idx[i+1];
		idx[i+1] = idx[end-1];
		idx[end-1] = tmp_i;

		i++;

		//
		// SELECT
		//
		if (i == k) {
			return dist[i];
		}
		else if (i < k){
			start = i+1;
		}
		else {
			end = i;
		}
	}

	return dist[start];
}


//! Sorts the arrays `dist` and `idx` in increasing order,
//!	based on the values of `dist`
/*!
	\param dist		The array of distances							[n-by-1]
	\param idx		The indexes on the initial array		[n-by-1]
	\param n			The length of the arrays						[scalar]
	-----------------------------------------------------------
	\return `void`
*/
void quickSort (double *dist, int *idx, int n) {

	if (n <= 1)
		return;

	//
	// PARTITION
	//
	double pivot = dist[n-1];

	int i = -1;

	for (int j = 0; j < n-1; j++) {
		if ( dist[j] <= pivot ) {
			i++;
			double tmp_d = dist[j];
			dist[j] = dist[i];
			dist[i] = tmp_d;

			int tmp_i = idx[j];
			idx[j] = idx[i];
			idx[i] = tmp_i;
		}
	}

	double tmp_d= dist[i+1];
	dist[i+1] = dist[n-1];
	dist[n-1] = tmp_d;

	int tmp_i = idx[i+1];
	idx[i+1] = idx[n-1];
	idx[n-1] = tmp_i;

	i++;

	quickSort(dist, idx, i);
	quickSort(dist+i+1, idx+i+1, n-i-1);

}


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
	double *x_sq = calloc(n,sizeof(double)),
				 *y_sq = calloc(n,sizeof(double));

	// Initializing to zero
  for(uint64_t i = 0; i < n*m; i++)
		D[i] = 0;

	// x_sq = sum(X.^2,2)
  for(uint64_t i = 0; i < n; i++)
		for(uint64_t j = 0; j < d; j++) x_sq[i] += pow( X[i*ldx+j], 2);

	// y_sq = sum(Y.^2,2)
  for(uint64_t i = 0; i < m; i++)
		for(int j = 0; j < d; j++) y_sq[i] += pow( Y[i*ldy+j], 2);

	//D += x_sq' + y_sq
  for(uint64_t i = 0; i < m; i++)
    for(uint64_t j = 0; j < n; j++) {
			D[i*ldD+j] = x_sq[j] + y_sq[i];
		}

	// D += -2*Y*X'
  product(D,Y,X,m,n,d);

	// Taking the square root
  for(uint64_t i = 0; i < n*m; i++) D[i] = sqrt(
																								(D[i] >= 0)? D[i] : -D[i]
																							);

	return;
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
		quickSelect(D+ldD*qp, idx, n, k);


		// Set the values in the result
		for(uint64_t cp = 0; cp < k; cp++) {

			result->nidx[ qp*ldr+cp ] = idx[ cp ];
			result->ndist[  qp*ldr+cp ] = D[ qp*ldD + cp  ];
		}

		// Sort the result
		quickSort(
					result->ndist +qp*ldr,
					result->nidx+qp*ldr,
					k
				);

	}

	free(D);
	free(idx);

	return *result;
}


//! Compute distributed all-kNN of points in X
knnresult distrAllkNN(double *X, int n, int d, int k) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Variable Decalration
	int P, pid,
			*idx = malloc(n*k*sizeof(int));
	double *dist = malloc(n*k*sizeof(double)),
				 *buffer = malloc(n*d*sizeof(double)),
				 *corpus = malloc(n*d*sizeof(double)),
				 *query = malloc(n*d*sizeof(double));
	MPI_Status stats[2];
	MPI_Request reqs[2];
	knnresult res, knn;

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initializing Varialbes
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &P);
	for(int i = 0; i < n*d; i++) {
		corpus[i] =	X[i];
		query[i] 	=	X[i];
		buffer[i] = X[i];
	}
	for(int i = 0; i < n*k; i++) {
			idx[ i ] 	= 	0;
			dist[ i ] = 	DBL_MAX;
	}

	for(int p = 0; p < P; p++) {

		MPI_Irecv(buffer, n*d, MPI_DOUBLE, prev(pid), TAG, MPI_COMM_WORLD, &reqs[0]);

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Computing the new kNN
		knn = kNN(corpus, query, n, n, d, k);
		// ________________________________ Updating the "global" kNN
		for(int i = 0; i < n; i++)
			for(int j = 0; j < k; j++) {
				knn.nidx[i*k+j] += offset(p);
				insert( dist+(i*k), idx+(i*k), knn.ndist[i*k+j], knn.nidx[i*k+j], k);
			}


		MPI_Isend(corpus, n*d, MPI_DOUBLE, next(pid), TAG, MPI_COMM_WORLD, &reqs[1]);

		MPI_Waitall(2, reqs, stats);

		for(int i = 0; i < n*d; i++) corpus[i] = buffer[i];

	}

	res.m = n;
	res.k = k;
	res.nidx = idx;
	res.ndist = dist;

	free(dist);
	free(buffer);
	free(corpus);
	free(query);
	free(idx);

	return res;

}
