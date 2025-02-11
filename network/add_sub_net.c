#include <types.h>
#include <network.h>
#include <connection.h>
#include <neuron.h>

#include <stdlib.h>

static
void
connect_from_set
(
	nn_uint		 count,
	nn_neuron	*from[count],
	nn_neuron	*to
)
{
	for(int i = 0; i < count; i++)
	{
		nn_connection_connect(from[i],to,nn_connection_default_weight(),false);
	}
}

static
void
connect_to_set
(
	nn_neuron	*from,
	nn_uint		 count,
	nn_neuron	*to[count]
)
{
	for(int i = 0; i < count; i++)
	{
		nn_connection_connect(from,to[i],nn_connection_default_weight(),false);
	}
}

void
nn_net_add_sub_net
(
	nn_net		*network,
	nn_net		*sub_net,
	nn_uint		 to_inputs_count,
	nn_neuron	*to_inputs[to_inputs_count],
	nn_uint		 from_outputs_count,
	nn_neuron	*from_outputs[from_outputs_count],
	nn_bool		 immutable
)
{
	for(int i = 0; i < sub_net->neurons_count; i++)
	{
		nn_neuron *neuron = nn_neuron_clone(sub_net->neurons[i]);

		nn_net_push_neuron(network,neuron);

		if(neuron->neuron_type == NN_INPUT_NEURON)
		{
			connect_from_set(to_inputs_count,to_inputs,neuron);
			neuron->neuron_type = NN_HIDDEN_NEURON;
		}

		if(neuron->neuron_type == NN_OUTPUT_NEURON)
		{
			connect_to_set(neuron,from_outputs_count,from_outputs);
			neuron->neuron_type = NN_HIDDEN_NEURON;

			continue; // don't immute to_connections
		}

		if(immutable)
		{
			for(int j = 0; j < neuron->to_connections_count; j++)
			{
				nn_connection_immutable(neuron->to_connections[j]);
			}
		}
	}
}
