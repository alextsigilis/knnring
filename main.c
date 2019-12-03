#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "knnring.h"

#define		prev(pid)			((P+pid-1)%P)
#define		offset(p)			(n*prev(pid-p))
#define 		TAG							1

int main (int argc, char *argv[]) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~ Declaring Variables
	int P, pid,
			n = atoi(argv[1]),
			d = atoi(argv[2]),
			k = atoi(argv[3]);
	double *X, *corpus;
	knnresult res, knn;
	MPI_Status stat;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &P);

	// ======================= Run the Sequential code
	if(P == 1) {
		X = malloc(n*d*P*sizeof(double));
		for(int i = 0; i < n*P*d; i++) X[i] = (double)(rand()) / RAND_MAX;

		kNN(X,X,n,n,d,k);

		return 0;

	}

	//			==================== MASTER
	if (pid == 0) {

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~ Read Input
		X = malloc(n*d*P*sizeof(double));
		for(int i = 0; i < n*P*d; i++) X[i] = (double)(rand()) / RAND_MAX;

		//  ~~~~~~~~~~~~~~~~~~~~~~~~~ Send chucks to each process
		for(int p = 0; p < P-1; p++) {
			MPI_Send(X+n*d*p, n*d, MPI_DOUBLE, p+1, TAG, MPI_COMM_WORLD);
		}

		// Last Chucnk is mine
		corpus = X + n*d*(P-1);
		knn = distrAllkNN(corpus, n, d, k);


		// Prepare result
		res.nidx = calloc(P*n*k, sizeof(int));
		res.ndist = calloc(P*n*k, sizeof(double));
		res.m = n*P;
		res.k = k;


		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Put mine into place
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < k; j++) {
				res.ndist[ (i+(P-1)*n)*k + j ] = knn.ndist[i*k+j];
				res.nidx[ (i+(P-1)*n)*k + j ] = knn.nidx[i*k+j];
			}
		}

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Collect result from
		//																				other processess
		for(int p = 0; p < P-1; p++) {
			MPI_Recv(knn.nidx, n*k, MPI_INT, p+1, TAG, MPI_COMM_WORLD, &stat);
			MPI_Recv(knn.ndist, n*k, MPI_DOUBLE, p+1, TAG, MPI_COMM_WORLD, &stat);

			for(int i = 0; i < n; i++) {
				for(int j = 0; j < k; j++) {
					res.ndist[ (i+p*n)*k + j ] = knn.ndist[i*k+j];
					res.nidx[ (i+p*n)*k + j ] = knn.nidx[i*k+j];
				}
			}

		}

	}
	//			====================== SLAVE
	else {

		corpus = malloc(n*d*sizeof(double));
		MPI_Recv(corpus, n*d, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &stat);

		knn = distrAllkNN(corpus, n, d, k);

		MPI_Send(knn.nidx, n*k, MPI_INT, 0, TAG, MPI_COMM_WORLD);
		MPI_Send(knn.ndist, n*k, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);

	}

	MPI_Finalize();

	return 0;
}
