#include <types.h>
#include <network.h>
#include <net_training_data.h>
#include <neuron.h>
#include <neuron_training_data.h>

#include "train.h"

void
nn_net_prepare_training
(
	nn_net	*network
)
{
	nn_network_training_data *td = nn_network_training_data_create();

	network->training_data = td;
}
