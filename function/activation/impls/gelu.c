#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
gelu
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

	(*outputs)[0] = (sum / 2) * (1 + erf(sum / sqrt(2)));
}

nn_float
gelu_phi
(
	nn_float	x
)
{
	return 0.5 * (1 + erf(x / sqrt(M_PI)));
}

void
gelu_prime
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

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = gelu_phi(sum) + sum * gelu_phi(sum);
	}
}

nn_activation_function *
nn_gelu_function
(
	void
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= gelu,
		.f_prime		= gelu_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
