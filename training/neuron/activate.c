#include <neuron.h>
#include <activation_function.h>
#include <training.h>

#include <stdlib.h>
#include <string.h>

#include "training_util.h"

void
nn_neuron_activate_training
(
	nn_net		*net,
	nn_neuron	*neuron
)
{
	if(net == 0 || neuron == 0)
	{
	}

	nn_network_training_data *td = net->training_data;

	nn_float from_values[neuron->from_connections_count];
	fill_from_connections(neuron,from_values);

	nn_float *output = malloc(neuron->to_connections_count * sizeof(nn_float));

	nn_activation_function_activate(neuron,from_values,output);

	nn_neuron_training_data *ntd = neuron->training_data;

	#if defined NN_USE_NEURON_MULTI_OUTPUT
	size_t s = neuron->to_connections_count * sizeof(nn_float);
	memcpy(neuron->output,output,s);
	ntd->outputs[td->r_input_num] = realloc(ntd->outputs[td->r_input_num],s);
	memcpy(ntd->outputs[td->r_input_num],output,s);
	#else
	neuron->output = *output;
	ntd->outputs[td->r_input_num] = neuron->output;
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */

	memcpy(ntd->from_outputs[td->r_input_num],from_values,sizeof(from_values));
}
