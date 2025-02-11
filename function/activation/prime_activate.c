#include <types.h>
#include <activation_function.h>

#include <string.h>

nn_status
nn_activation_function_prime_activate
(
	nn_neuron	*neuron,
	nn_float	*inputs,
	nn_float	*outputs
)
{
	nn_activation_function *af = neuron->activation_function;

	nn_uint output_count = 0;
	nn_float *outs = 0;

	neuron->activation_function->f_prime(
		neuron->from_connections_count,
		inputs,
		af->coefficients_count,
		af->coefficients,
		neuron,
		af->extra,
		&output_count,
		&outs
	);

	if(output_count != neuron->to_connections_count)
	{
	}

	memcpy(outputs,outs,neuron->to_connections_count * sizeof(nn_float));

	return NN_SUCCESS;
}
