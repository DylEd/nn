#include <types.h>
#include <network.h>
#include <net_training_data.h>
#include <neuron.h>
#include <neuron_training_data.h>

#include "train.h"

void
nn_net_prepare_training_round
(
	nn_net	*network
)
{
	nn_network_training_data *td = network->training_data;
	for(int i = 0; i < network->neurons_count; i++)
	{
		nn_neuron_prepare_training(network->neurons[i],td->r_input_count);
	}
}
