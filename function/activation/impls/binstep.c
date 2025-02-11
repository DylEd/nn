#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
binstep
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

	(*outputs)[0] = sum < coefficients[0] ? 0.0 : 1.0;
}

void
binstep_prime
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

	memset(*outputs,0,inputs_count * sizeof(nn_float));
}

nn_activation_function *
nn_binstep_function
(
	nn_float	t
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = t;

	*f = (nn_activation_function) {
		.f			= binstep,
		.f_prime		= binstep_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
