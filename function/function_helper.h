#ifndef __FUNCTION_HELPER
#define __FUNCTION_HELPER

#include <types.h>
#include <functions.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define new_func malloc(sizeof(nn_activation_function))
#define new_loss_func malloc(sizeof(nn_loss_function))

#define new_coe(n) malloc((n) * sizeof(nn_float))

static
inline
nn_float
sum_arr
(
	nn_uint		n,
	nn_float	arr[n]
)
{
	nn_float s = 0.0;
	for(int i = 0; i < n; i++)
	{
		s += arr[i];
	}
	return s;
}

#endif /* __FUNCTION_HELPER */
