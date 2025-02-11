#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
identity
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

	(*outputs)[0] = sum;
}

void
identity_prime
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
nn_identity_function
(
	void
)
{
	nn_activation_function *f = new_func;

	*f = (nn_activation_function) {
		.f		= identity,
		.f_prime	= identity_prime,
	};

	return f;
}
