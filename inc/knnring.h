/*
 * File		: heap.h

 * Title	: Heap

 * Short	: Routines for heap implemetation

 * Long 	: -

 * Author : Αλέξανδρος Τσιγγίλης

 * Date		: 29 November 2019

*/

#ifndef __KNNRING_H__
#define __KNNRING_H__

// Definition of the kNN result struct
typedef struct knnresult{
	int * nidx; 					//!< Indices (0-based) of nearest neighbors 		[m-by-k]
	double * ndist; 			//!< Distance of nearest neighbors 							[m-by-k]
	int m; 								//!< Number of query points 										[scalar]
	int k; 								//!< Number of nearest neighbors 								[scalar]
} knnresult;


//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
	\param X 					Corpus data points 					[n-by-d]
	\param Y 					Query data points 					[m-by-d]
	\param n 					Number of corpus points 		[scalar]
	\param m 					Number of query points 			[scalar]
	\param d 					Number of dimensions 				[scalar]
	\param k 					Number of neighbors 				[scalar]
	------------------------------------------------------
	\return 	The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);


//! Compute distributed all-kNN of points in X
/*!
	\param X 			Data points 							[n-by-d]
	\param n 			Number of data points 		[scalar]
	\param d 			Number of dimensions 			[scalar]
	\param k 			Number of neighbors 			[scalar]
	--------------------------------------------------
	\return The kNN result
*/
knnresult distrAllkNN(double * X, int n, int d, int k);



/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%% */
#include <float.h>

#define		parent(i)				(int) floor((double)(i)/2)
#define		left(i)							2*i+1
#define		right(i)						2*i+2
#define 	insert(a,b,c,d,e)		decHeapKey(a,b,c,d,e)


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
void decHeapKey(double *dist, int *idx, int n, double v, int id){

	int i = n-1;

	if( v > dist[i] )
		return;


	dist[i] = v;
	idx[i] = id;

	while( i > 0 && dist[parent(i)] > dist[i] ){
		double tmp_d = dist[i];
		dist[i] = dist[parent(i)];
		dist[parent(i)] = tmp_d;

		int tmp_i = idx[i];
		idx[i] = idx[parent(i)];
		idx[parent(i)] = tmp_i;

		i = parent(i);
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

/* %%%%%%%%%%%%%%%% END of MY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%% */

#endif // __KNN_RING_H__
