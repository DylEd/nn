#ifndef __NEURON_UTIL
#define __NEURON_UTIL

#include <neuron.h>
#include <stdlib.h>
#include <string.h>

static
inline
nn_neuron *
new_neuron
(
	void
)
{
	return malloc(sizeof(nn_neuron));
}

static
void
net_push_neuron
(
	nn_net		*net,
	nn_neuron	*neuron
)
{
	net->neurons = realloc(
		net->neurons,
		(net->neurons_count + 1) * sizeof(nn_neuron *)
	);

	net->neurons[net->neurons_count] = neuron;

	net->neurons_count++;
}

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

#endif /* __NEURON_UTIL */
