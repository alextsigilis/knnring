/*
 * File		: heap.h

 * Title	: Heap

 * Short	: Routines for heap implemetation

 * Long 	: -

 * Author : Αλέξανδρος Τσιγγίλης

 * Date		: 29 November 2019

 */

#ifndef __HEAP_H__
#define __HEAP_H__

#include <float.h>

#define		parent(i)				(int) floor((double)(i)/2)
#define		left(i)							2*i+1
#define		right(i)						2*i+2
#define 	insert(a,b,c,d,e)		decHeapKey(a,b,c,d,e)


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





#endif	/* __HEAP_H__ */
