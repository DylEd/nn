#include <types.h>
#include <network.h>
#include <neuron.h>

void
nn_net_run
(
	nn_net		*network,
	nn_float	*input
)
{
	network->network_input = input;

	for(int i = 0; i < network->neurons_count; i++)
	{
		nn_neuron_activate(network,network->neurons[i]);
	}
}
