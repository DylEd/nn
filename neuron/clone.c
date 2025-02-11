#include <types.h>
#include <neuron.h>

#include <stdlib.h>
#include <string.h>

nn_neuron *
nn_neuron_clone
(
	nn_neuron	*neuron
)
{
	nn_neuron *n = malloc(sizeof(nn_neuron));

	memcpy(n,neuron,sizeof(nn_neuron));

	return n;
}
