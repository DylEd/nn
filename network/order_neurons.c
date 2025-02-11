#include <types.h>
#include <network.h>
#include <neuron_training_data.h>

#include <stdlib.h>
#include <string.h>

static
int
neuron_compare_id
(
	const void	*aa,
	const void	*bb
)
{
	nn_neuron *a = (nn_neuron *)aa;
	nn_neuron *b = (nn_neuron *)bb;

	return a->neuron_type - b->neuron_type;
}

static
int
neuron_compare_type
(
	const void	*aa,
	const void	*bb
)
{
	nn_neuron *a = (nn_neuron *)aa;
	nn_neuron *b = (nn_neuron *)bb;

	return a->id - b->id;
}

static
void
sort_hidden_neurons
(
	nn_net	*network
)
{
	int hiddens_count = network->hidden_neurons_count;
	nn_neuron *hiddens[hiddens_count];

	for(int i = 0; i < network->hidden_neurons_count; i++)
	{
		hiddens[i] = network->hidden_neurons[i];
	}

	int net_hidden_index = 0;
	int prev_count = -1;
	int hiddens_index = 0;
	while(hiddens_count != 0)
	{
		bool insert = true;
		nn_neuron *neuron = hiddens[hiddens_index];

		for(int i = 0; i < neuron->from_connections_count; i++)
		{
			if(neuron->from_connections[i]->from_neuron->neuron_type != NN_HIDDEN_NEURON)
			{
				continue;
			}

			nn_uint fnid = neuron->from_connections[i]->from_neuron->id;

			bool found = false;

			// verify that fn comes before neuron in hiddens
			for(int j = 0; j < net_hidden_index; j++)
			{
				nn_uint pnid = hiddens[j]->id;

				if(fnid == pnid)
				{
					found = true;
					break;
				}
			}

			if(!found)
			{
				insert = false;
				break;
			}
		}

		if(insert)
		{
			network->hidden_neurons[net_hidden_index] = hiddens[hiddens_index];
			net_hidden_index++;

			hiddens[hiddens_index] = hiddens[hiddens_count - 1];
			hiddens_count--;
			hiddens_index %= hiddens_count;
		}
		else
		{
			hiddens_index = (hiddens_index + 1) % hiddens_count;

			if(hiddens_index == 0)
			{
				if(hiddens_count == prev_count)
				{
					printf("there appears to be a cyclic link\n");
					exit(1);
				}
				prev_count = hiddens_count;
			}
		}
	}

	memcpy(network->hidden_neurons,hiddens,sizeof(hiddens));
}

void
nn_network_order_neurons
(
	nn_net	*network
)
{
	network->input_neurons = network->neurons;
	network->hidden_neurons = network->input_neurons + network->input_neurons_count;
	network->output_neurons = network->hidden_neurons + network->hidden_neurons_count;
	network->recurrent_neurons = network->output_neurons + network->output_neurons_count;

	// sort by neuron_type:
	//	NN_INPUT_NEURON < NN_HIDDEN_NEURON < NN_OUTPUT_NEURON < NN_RECURRENT_NEURON
	qsort(network->neurons,sizeof(nn_neuron *),network->neurons_count * sizeof(nn_neuron *),neuron_compare_type);

	sort_hidden_neurons(network);

	// enumerate the order of each neuron type
	for(int i = 0; i < network->input_neurons_count; i++)
	{
		network->input_neurons[i]->order = i;
	}

	for(int i = 0; i < network->hidden_neurons_count; i++)
	{
		network->hidden_neurons[i]->order = i;
	}

	for(int i = 0; i < network->output_neurons_count; i++)
	{
		network->output_neurons[i]->order = i;
	}

	for(int i = 0; i < network->recurrent_neurons_count; i++)
	{
		network->recurrent_neurons[i]->order = i;
	}
}
