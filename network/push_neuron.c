#include <types.h>
#include <network.h>

#include <stdlib.h>

void
nn_net_push_neuron
(
	nn_net		*network,
	nn_neuron	*neuron
)
{
	network->neurons_count++;

	network->neurons = realloc(network->neurons,sizeof(nn_neuron *) * network->neurons_count);

	network->neurons[network->neurons_count - 1] = neuron;

	switch(neuron->neuron_type)
	{
		case NN_HIDDEN_NEURON:
			network->hidden_neurons_count++;
			break;
		case NN_INPUT_NEURON:
			network->input_neurons_count++;
			break;
		case NN_OUTPUT_NEURON:
			network->output_neurons_count++;
			break;
		case NN_RECURRENT_NEURON:
			network->recurrent_neurons_count++;
			break;
	}
}
