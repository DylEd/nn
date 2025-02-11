#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

static
inline
nn_float
avg_arr
(
	nn_uint		xn,
	nn_float	x[xn]
)
{
	nn_float sum = 0.0;

	for(int i = 0; i < xn; i++)
	{
		sum += x[i];
	}

	return sum / xn;
}

void
avg_pool
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	(*outputs)[0] = avg_arr(inputs_count,inputs);
}

void
avg_pool_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = 1.0 / inputs_count;
	}
}

nn_activation_function *
nn_avg_pool_function
(
	void
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= avg_pool,
		.f_prime		= avg_pool_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
