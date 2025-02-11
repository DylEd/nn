#include <types.h>
#include <functions.h>

#include "../../function_helper.h"

void
smht
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

	(*outputs)[0] = (exp(coefficients[0] * sum) - exp(coefficients[1] * -sum)) / (exp(coefficients[2] * sum) + exp(coefficients[3] * -sum));
}

void
smht_prime
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

	nn_float ea = exp(coefficients[0] * sum);
	nn_float eb = exp(coefficients[1] * -sum);
	nn_float ec = exp(coefficients[2] * sum);
	nn_float ed = exp(coefficients[3] * -sum);

	nn_float y = (ea - eb) / (ec + ed);

	nn_float y_p = (coefficients[0] * ea - coefficients[1] * eb - y * (coefficients[2] * ec - coefficients[3]*ed)) / (ec + ed);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = y_p;
	}
}

nn_activation_function *
nn_smht_function
(
	nn_float	a,
	nn_float	b,
	nn_float	c,
	nn_float	d
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 4;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;
	coe[1] = b;
	coe[2] = c;
	coe[3] = d;

	*f = (nn_activation_function) {
		.f			= smht,
		.f_prime		= smht_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
