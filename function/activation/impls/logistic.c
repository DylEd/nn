#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
logistic
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

	(*outputs)[0] = 1/(1 + exp(coefficients[0] * -sum));
}

void
logistic_prime
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

	nn_float l = 1/(1 + exp(-sum));

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = coefficients[0] * l * (1-l);
	}
}

nn_activation_function *
nn_logistic_function
(
	nn_float	a
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;

	*f = (nn_activation_function) {
		.f			= logistic,
		.f_prime		= logistic_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
