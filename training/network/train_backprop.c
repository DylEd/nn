#include <types.h>
#include <network.h>
#include <neuron.h>
#include <training.h>
#include <loss_function.h>

#include "train.h"

void
nn_net_train
(
	nn_net			 *network,
	nn_uint			  inputs_count,
	nn_float		 *inputs[inputs_count],	// [m->input_neurons_count]
	nn_uint			  r_inputs[inputs_count],
	nn_float		 *expected_outputs[inputs_count], // [m->output_neurons_count]
	nn_uint			  epochs,
	nn_float		  training_rate,
	nn_loss_function	  loss_function,
	nn_uint			(*callback_function)
				(
					nn_net		*m,
					nn_float	 loss,
					nn_uint		 epoch
				)
)
{
	nn_float output[network->output_neurons_count];

	nn_network_training_data *td = nn_network_training_data_create();
	network->training_data = td;

	for(int input = 0; input < inputs_count; input++)
	{
		// for the purposes of parallelization (not yet implemented)
		//	... or an array of semaphores
		//	.. probably add in nn_training_data
		//nn_uint index[m->neurons_count] = {0};
		// index[n] = epoch of most recent update
		// if a neuron depends on a neuron that has not updated index to the current epoch, wait

		// might be able to use call_once (<threads.h>)
		// once_flag flags[m->neurons_count] = {ONCE_FLAG_INIT};
		// in forward pass & backpropagation
		//	call_once(&flags[neuron],nn_neuron_activate-training);	// note, though, that the function is void (*)(void)

		network->network_input = inputs[input];

		td->r_input_num = 0;
		td->r_input_count = r_inputs[input];
		td->outputs = output;
		td->expected_outputs = expected_outputs[input];

		nn_net_prepare_training_round(network);

		for(int epoch = 0; epoch < epochs; epoch++)
		{
			int neuron = 0;

			// forward pass
			for(td->r_input_num = 0; td->r_input_num < td->r_input_count; td->r_input_num++)
			{
				for(; neuron < network->neurons_count; neuron++)
				{
					nn_neuron_activate_training(network,network->neurons[neuron]);
				}
			}

			// make output convenient
			for(int o = 0; o < network->output_neurons_count; o++)
			{
				#if defined NN_USE_NEURON_MULTI_OUTPUT
				output[o] = network->output_neurons[o]->output[0];
				#else
				output[o] = network->output_neurons[o]->output;
				#endif /* NN_USE_NEURON_MULTI_OUTPUT */
			}

			// callback / condition checking
			nn_float *loss = 0;
			nn_uint loss_count = 0;
			//nn_float loss = td->loss_function.f(m->output_neurons_count,output,td->expected_outputs);
			nn_loss_function_activate(
				td->loss_function,
				network->output_neurons_count,
				output,
				td->expected_outputs,
				&loss_count,
				&loss
			);
			if(callback_function != 0 && callback_function(network,loss[0],epoch))
			{
				break;
			}
			nn_loss_function_prime_activate(
				td->loss_function,
				network->output_neurons_count,
				td->outputs,
				td->expected_outputs,
				&network->output_neurons_count,
				&td->loss_deltas
			);

			// backpropagation
			for(; td->r_input_num >= 0; td->r_input_num--)
			{
				for(; neuron >= 0; neuron--)
				{
					nn_neuron_backprop_training(network,network->neurons[neuron]);
				}
			}

			// update weights
			for(neuron = network->input_neurons_count + network->recurrent_neurons_count; neuron < network->neurons_count; neuron++)
			{
				nn_neuron *n = network->neurons[neuron];
				nn_neuron_training_data *npd = n->training_data;

				for(int connection = 0; connection < n->from_connections_count; connection++)
				{
					nn_connection *c = n->from_connections[connection];

					if(c->immutable)
					{
						continue;
					}

					for(int r_input = 0; r_input < td->r_input_count; r_input++)
					{
						nn_float delta = npd->deltas[r_input][connection];
						nn_float output;
						#if defined NN_USE_NEURON_MULTI_OUTPUT
						output = npd->outputs[r_input][connection];
						#else
						output = npd->outputs[r_input];
						#endif /* NN_USE_NEURON_MULTI_OUTPUT */

						c->weight -= training_rate * delta * output;
					}
				}
			}
		}

		nn_network_training_data_free(td);
	}
}
