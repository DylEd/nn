#include <types.h>
#include <network.h>

#include <stdlib.h>

nn_net *
nn_net_create
(
	void
)
{
	nn_net *net = malloc(sizeof(nn_net));

	nn_neuron **neurons = malloc(0);

	*net = (nn_net) {
		.neurons = neurons,
	};

	return net;
}
