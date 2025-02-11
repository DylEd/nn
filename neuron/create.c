#include <neuron.h>

#include "neuron_util.h"

#include <stddef.h>

nn_neuron *
nn_neuron_create_from_struct
(
	nn_net			*net,
	nn_neuron		 neuron_spec
)
{
	static nn_uint static_id = 0;

	if(neuron_spec.activation_function == 0)
	{
		neuron_spec.activation_function = nn_default_activation_function;
	}

	nn_neuron *n = new_neuron();

	n->activation_function = neuron_spec.activation_function;
	n->neuron_type = neuron_spec.neuron_type;
	#if defined NN_USE_NEURON_MULTI_OUTPUT
	n->repeat_type = neuron_spec.repeat_type;
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */
	n->extra = neuron_spec.extra;

	// overwrite const field n->id
	*(((nn_uint *) &n) + offsetof(nn_neuron,id)) = static_id;

	n->from_connections = malloc(0);
	n->from_connections_count = 0;
	n->to_connections = malloc(0);
	n->to_connections_count = 0;
	#if defined NN_USE_NEURON_MULTI_OUTPUT
	n->output = malloc(0);
	n->ouput_count = 0;
	#else
	n->output = 0;
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */

	static_id++;

	net_push_neuron(net,n);

	return n;
}
