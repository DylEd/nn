#include <types.h>
#include <loss_function.h>

#include <stdlib.h>
#include <stdio.h>

nn_status
nn_loss_function_activate
(
	nn_loss_function	 *function,
	nn_uint			  inputs_count,
	nn_float		  inputs[inputs_count],
	nn_float		  expected[inputs_count],
	nn_uint			 *outputs_count,
	nn_float		**outputs
)
{
	if(function == 0 || inputs == 0 || expected ==0)
	{
	}

	if(outputs == 0 || outputs_count == 0)
	{
	}

	if(*outputs == 0)
	{
		*outputs = malloc(inputs_count * sizeof(nn_float));
	}

	function->f(
		inputs_count,
		inputs,
		expected,
		function->coefficients_count,
		function->coefficients,
		function->extra,
		outputs_count,
		outputs
	);

	if(*outputs_count != 1)
	{
		puts("");
		exit(1);
	}

	return NN_SUCCESS;
}
