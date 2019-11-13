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
#include <gsl/gsl_cblas.h>	
#include "knnring.h"

#define IDX(d,i,j) 		(d*i)+j


knnresult kNN(double *X, double *Y, int n, int m, int d, int k) {

	knnresult *tmp = malloc(sizeof(knnresult));

	return *tmp;
}
