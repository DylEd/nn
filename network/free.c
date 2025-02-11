#include <types.h>
#include <network.h>
#include <neuron.h>
#include <net_training_data.h>

#include <stdlib.h>

void
nn_net_free
(
	nn_net	*net
)
{
	for(int i = 0; i < net->neurons_count; i++)
	{
		nn_neuron_free(net,net->neurons[i]);
	}

	free(net->neurons);

	nn_network_training_data_free(net->training_data);

	free(net);
}
