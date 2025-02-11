#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

static
inline
nn_float
max_arr
(
	nn_uint		xn,
	nn_float	x[xn]
)
{
	nn_float m = x[0];

	for(int i = 1; i < xn; i++)
	{
		if(m < x[i])
		{
			m = x[i];
		}
	}

	return m;
}

void
max_pool
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

	(*outputs)[0] = max_arr(inputs_count,inputs);
}

void
max_pool_prime
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

	nn_float m = max_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = m == inputs[i] ? 1.0 : 0.0;
	}
}

nn_activation_function *
nn_max_pool_function
(
	void
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= max_pool,
		.f_prime		= max_pool_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
