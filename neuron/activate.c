#include <neuron.h>
#include <network.h>
#include <activation_function.h>

#include "neuron_util.h"

#include <string.h>

void
nn_neuron_activate
(
	nn_net		*net,
	nn_neuron	*neuron
)
{
	if(net == 0 || neuron == 0)
	{
	}

	if(neuron->neuron_type == NN_INPUT_NEURON)
	{
		neuron->output = nn_get_input(net,neuron->order);
		return;
	}

	nn_float from_values[neuron->from_connections_count];
	fill_from_connections(neuron,from_values);

	nn_float *output = malloc(neuron->to_connections_count * sizeof(nn_float));

	nn_activation_function_activate(neuron,from_values,output);

	#if defined NN_USE_NEURON_MULTI_OUTPUT
	memcpy(neuron->output,output,neuron->to_connections_count * sizeof(nn_float));
	#else
	neuron->output = *output;
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */
}
