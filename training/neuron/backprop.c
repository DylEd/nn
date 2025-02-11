#include <types.h>
#include <training.h>
#include <activation_function.h>

#include <string.h>

#include "training_util.h"

void
nn_neuron_backprop_training
(
	nn_net		*net,
	nn_neuron	*neuron
)
{
	if(net == 0 || neuron == 0)
	{
	}

	nn_neuron_training_data *td = neuron->training_data;
	nn_network_training_data *ntd = net->training_data;

	nn_uint r_num = ntd->r_input_num;

	nn_float delta = 0.0;

	// calculate delta
	switch(neuron->neuron_type)
	{
		case NN_INPUT_NEURON:
			/* do nothing */
			return;
		case NN_OUTPUT_NEURON:
			// only consider error of last recurrent input (final unwrap)
			if(r_num != ntd->r_input_count)
			{
				memset(td->deltas[r_num],0,neuron->from_connections_count * sizeof(nn_float));
				/*
					XXX: perhaps make a net function/field
						nn_uint
						nn_net_r_error
						(
							nn_net		*net,
							nn_neuron	*neuron
						);
						or
						ntd->loss_deltas[r_num][find_output_index(neuron,net)];
				*/
				return;
			}
			delta = ntd->loss_deltas[find_output_index(neuron,net)];
			//delta = ntd->loss_deltas[td->type_order];
			break;
		case NN_RECURRENT_NEURON:
			if(r_num == ntd->r_input_count)
			{
				memset(td->deltas[r_num],0,neuron->from_connections_count * sizeof(nn_float));
			}
			/* fall through */
		default: /* NN_HIDDEN_NEURON */
			for(int i = 0; i < neuron->to_connections_count; i++)
			{
				nn_connection *c = neuron->to_connections[i];
				nn_neuron *tc = c->to_neuron;
				nn_neuron_training_data *ttd = tc->training_data;
				delta += c->weight * ttd->deltas[r_num][find_from_index(neuron,tc)];
			}
			break;
	}

	if(delta == 0.0)
	{
		memset(td->deltas[r_num],0,neuron->from_connections_count * sizeof(nn_float));
		return;
	}

	nn_float *from_values = neuron->training_data->from_outputs[r_num];

	nn_float *gradient = malloc(neuron->to_connections_count * sizeof(nn_float));
	nn_activation_function_prime_activate(neuron,from_values,gradient);

	td->deltas[r_num] = gradient;
}
