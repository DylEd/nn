#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
linear
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

	nn_float sum = sum_arr(inputs_count,inputs);

	*outputs[0] = coefficients[0] * sum + coefficients[1];
}

void
linear_prime
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
		(*outputs)[i] = coefficients[0];
	}
}

nn_activation_function *
nn_linear_function
(
	nn_float	m,
	nn_float	b
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 2;

	nn_float *coe = new_coe(coe_num);

	coe[0] = m;
	coe[1] = b;

	*f = (nn_activation_function) {
		.f			= linear,
		.f_prime		= linear_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
