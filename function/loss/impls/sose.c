#include <types.h>
#include <function.h>

#include "../../function_helper.h"

void
sose
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_float	  expected[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = 0.0;
	for(int i = 0; i < coefficients_count; i++)
	{
		sum += pow(inputs[i] - expected[i],2);
	}

	(*outputs)[0] = coefficients[0] * sum;
}

void
sose_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_float	  expected[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	for(int i = 0; i < coefficients_count; i++)
	{
		(*outputs)[i] = coefficients[0] * 0.5 * fabs(inputs[i] - expected[i]);
	}
}

nn_loss_function *
nn_sose_loss_function
(
	nn_float	a
)
{
	nn_loss_function *f = new_loss_func;

	f = new_loss_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;

	*f = (nn_loss_function) {
		.f			= sose,
		.f_prime		= sose_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
