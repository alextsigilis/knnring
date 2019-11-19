/*

 * File   : knnring_sequential.c

 * Title  : Sequential kNN Ring

 * Short  : A brute force sollution to full k nearest neighbors.

 * Long   : -

 * Author : Αλέξανδρος Τσιγγίλης

 * Date   : 19 November 2019

*/

#ifndef __UTILS_H__
#define __UTILS_H__

//! Computes the product -2 * A*B' and assigns it to C
/*!
  \param C      The resuting matrix       [n-by-m]
  \param A      The first matrix          [n-by-d]
  \param B      The second matrix         [m-by-d]
  \param n      Rows of A                 [scalar]
  \param m      Rows of B                 [scalar]
  \param d      Collumns of A and B       [scalar]
*/
void product(double *C, double *A, double *B, int n, int m, int d);



//! Computes the distance of every point in X to every point in Y
/*!
	\param D			The result matrix				[m-by-d]
	\param X			The corpus points				[n-by-d]
	\param Y			The query points				[m-by-d]
	\param n			The number of						[scalar]
									corpus points
	\param m			The number of						[scalar]
									query points
	\param d			The number of dimetions	[scalar]
*/
void compute_distances (double *D, double *X, double *Y, int n, int m, int d);



//! Partitions the elemets of X
/*!
	\param X		The elemets to be partitioned		[n-by-1]
	\param idx	The indexes of the elements			[n-by-1]
	\paraam n		The number of elements					[scalar]
	----------------------------------------------------
	\return the possition of the pivot
*/
int qPartition(double *X, int *idx, int n);


//! "Selects" the k-th smallest element in X,
//! (it also partitions the array in smaller and bigger elements)
/*!
	\param X			The set of elements			[n-by-1]
	\param idx		The indexes of the 			[n-by-1]
											elements
	\param n			The number of elements	[scalar]
	\param k			The elements to be			[scalar]
									selected
	---------------------------------------------
	\return the k-th smallest elemet
*/
double qSelect(double *X, int *idx, int n, int k);

#endif //__UTILS_H__
