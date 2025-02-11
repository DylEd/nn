#ifndef __TRAINING_HELPER
#define __TRAINING_HELPER

#include <types.h>

static
void
fill_from_connections
(
	nn_neuron	*neuron,
	nn_float	*dest
)
{
	for(int i = 0; i < neuron->from_connections_count; i++)
	{
		nn_float weight = neuron->from_connections[i]->weight;
		nn_neuron *from_neuron = neuron->from_connections[i]->from_neuron;

		dest[i] = weight * from_neuron->output;
	}
}

static
int
find_from_index
(
	nn_neuron	*neuron,
	nn_neuron	*to_neuron
)
{
	nn_uint nid = neuron->id;

	for(int i = 0; i < to_neuron->from_connections_count; i++)
	{
		nn_uint tid = to_neuron->from_connections[i]->from_neuron->id;

		if(tid == nid)
		{
			return i;
		}
	}

	return -1;
}

static
int
find_to_index
(
	nn_neuron	*neuron,
	nn_neuron	*from_neuron
)
{
	nn_uint nid = neuron->id;

	for(int i = 0; i < from_neuron->to_connections_count; i++)
	{
		nn_uint tid = from_neuron->to_connections[i]->to_neuron->id;

		if(tid == nid)
		{
			return i;
		}
	}

	return -1;
}

static
int
find_output_index
(
	nn_neuron	*neuron,
	nn_net		*net
)
{
	nn_uint nid = neuron->id;

	for(int i = 0; i < net->output_neurons_count; i++)
	{
		nn_uint tid = net->output_neurons[i]->id;

		if(tid == nid)
		{
			return i;
		}
	}

	return -1;
}

#endif /* __TRAINING_HELPER */
