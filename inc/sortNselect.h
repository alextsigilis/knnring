#ifndef __SORT_N_SELECT__H__
#define __SORT_N_SELECT__H__

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

#endif // __SORT_N_SELECT__H__
