#include "nn_types.h"
#include "nn_private.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "include/vararr.h"

nn_net *
nn_create_net
(
	void
)
{
	nn_net *m = malloc(sizeof(nn_net));

	*m = (nn_net) {0};

	m->neurons_va = va_new(sizeof(nn_neuron *));

	return m;
}

void
nn_free_net
(
	nn_net	*m
)
{
	for(int i = 0; i < m->neurons_count; i++)
	{
		nn_free_neuron(m->neurons[i]);
	}
	va_free(m->neurons_va);
	nn_free_training_data(m->training_info);
}

void
nn_net_run
(
	nn_net		*m,
	nn_float	*input
)
{
	m->input = input;

	for(int i = 0; i < m->neurons_count; i++)
	{
		nn_neuron_activate(m,m->neurons[i]);
	}
}

void
nn_net_train
(
	nn_net			 *m,
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
	nn_float output[m->output_neurons_count];

	m->training_info = nn_create_training_data();
	nn_training_data *td = m->training_info;

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

		m->input = inputs[input];

		td->r_input_num = 0;
		td->r_input_count = r_inputs[input];
		td->outputs = output;
		td->expected_outputs = expected_outputs[input];

		nn_private_neuron_data_prep_training(m);

		for(int epoch = 0; epoch < epochs; epoch++)
		{
			int neuron = 0;

			// forward pass
			for(td->r_input_num = 0; td->r_input_num < td->r_input_count; td->r_input_num++)
			{
				for(; neuron < m->neurons_count; neuron++)
				{
					nn_neuron_activate_training(m,m->neurons[neuron]);
				}
			}

			// make output convenient
			for(int o = 0; o < m->output_neurons_count; o++)
			{
				output[o] = m->output_neurons[o]->outputs[0];
			}

			// callback / condition checking
			nn_float *loss = 0;
			nn_uint loss_count = 0;
			//nn_float loss = td->loss_function.f(m->output_neurons_count,output,td->expected_outputs);
			nn_activate_loss_function(
				td->loss_function,
				m->output_neurons_count,
				output,
				td->expected_outputs,
				&loss_count,
				&loss
			);
			if(callback_function != 0 && callback_function(m,loss[0],epoch))
			{
				break;
			}
			nn_activate_loss_function_prime(
				m->training_info->loss_function,
				m->output_neurons_count,
				m->training_info->outputs,
				m->training_info->expected_outputs,
				&m->output_neurons_count,
				&m->training_info->loss_deltas
			);

			// backpropagation
			for(; td->r_input_num >= 0; td->r_input_num--)
			{
				for(; neuron >= 0; neuron--)
				{
					nn_neuron_backprop_training(m,m->neurons[neuron]);
				}
			}

			// update weights
			for(neuron = m->input_neurons_count + m->recurrent_neurons_count; neuron < m->neurons_count; neuron++)
			{
				nn_neuron *n = m->neurons[neuron];
				nn_private_neuron_data *npd = n->private_data;

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
						nn_float output = npd->outputs[r_input][connection];

						c->weight -= training_rate * delta * output;
					}
				}
			}
		}

		nn_free_training_data(td);
	}
}

int
neuron_compare_type
(
	const void	*aa,
	const void	*bb
)
{
	nn_neuron *a = (nn_neuron *)aa;
	nn_neuron *b = (nn_neuron *)bb;

	return a->neuron_type - b->neuron_type;
}

int
neuron_compare_id
(
	const void	*aa,
	const void	*bb
)
{
	nn_neuron *a = (nn_neuron *)aa;
	nn_neuron *b = (nn_neuron *)bb;

	nn_float aid = a->private_data->id;
	nn_float bid = b->private_data->id;

	return aid - bid;
}

void
sort_hidden_neurons
(
	nn_net	*m
)
{
	int hiddens_count = m->hidden_neurons_count;
	nn_neuron *hiddens[hiddens_count];

	for(int i = 0; i < m->hidden_neurons_count; i++)
	{
		hiddens[i] = m->hidden_neurons[i];
	}

	int net_base = m->input_neurons_count + m->recurrent_neurons_count;
	int net_hidden_index = 0;
	int prev_count = -1;
	int hiddens_index = 0;
	while(hiddens_count != 0)
	{
		bool insert = true;
		nn_neuron *n = hiddens[hiddens_index];

		// go through each neuron connection to n
		for(int i = 0; i < n->from_connections_count; i++)
		{
			nn_private_neuron_data *fnpd = n->from_connections[i]->from_neuron->private_data;
			nn_uint fnid = fnpd->id;

			bool found = false;

			// verify that fn comes before n in m->neurons;
			// TODO: may want to reduces to only hidden neurons
			for(int j = 0; j < net_base + net_hidden_index; j++)
			{
				nn_private_neuron_data *pnpd = m->neurons[j]->private_data;
				nn_uint pnid = pnpd->id;

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
			m->hidden_neurons[net_hidden_index] = hiddens[hiddens_index];
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
}

void
nn_order_neurons
(
	nn_net	*m
)
{
	m->input_neurons = m->neurons;
	m->recurrent_input_neurons = m->input_neurons + m->input_neurons_count;
	m->hidden_neurons = m->recurrent_input_neurons + m->recurrent_neurons_count;
	m->output_neurons = m->hidden_neurons + m->hidden_neurons_count;
	m->recurrent_output_neurons = m->output_neurons + m->output_neurons_count;

	// sort by neuron_type:
	//	NN_INPUT_NEURON < NN_RECURRENT_INPUT_NEURON < NN_HIDDEN_NEURON < NN_OUTPUT_NEURON < NN_RECURRENT_OUTPUT_NEURON
	qsort(m->neurons,sizeof(nn_neuron *),m->neurons_count * sizeof(nn_neuron *),neuron_compare_type);

	sort_hidden_neurons(m);

	for(int i = 0; i < m->input_neurons_count; i++)
	{
		m->input_neurons[i]->private_data->order = i;
	}

	for(int i = 0; i < m->hidden_neurons_count; i++)
	{
		m->hidden_neurons[i]->private_data->order = i;
	}

	for(int i = 0; i < m->output_neurons_count; i++)
	{
		m->output_neurons[i]->private_data->order = i;
	}

	// connected recurrent neurons have the same id
	qsort(m->recurrent_input_neurons,m->recurrent_neurons_count,sizeof(nn_neuron *),neuron_compare_id);
	qsort(m->recurrent_output_neurons,m->recurrent_neurons_count,sizeof(nn_neuron *),neuron_compare_id);
	for(int i = 0; i < m->recurrent_neurons_count; i++)
	{
		m->recurrent_input_neurons[i]->private_data->order = i;
		m->recurrent_output_neurons[i]->private_data->order = i;
	}
}

void
nn_add_sub_net
(
	nn_net		*m,
	nn_net		*sub_net,
	nn_uint		 to_inputs_count,
	nn_neuron	 to_inputs[to_inputs_count],
	nn_uint		 from_outputs_count,
	nn_neuron	 from_outputs[from_outputs_count],
	nn_bool		 immutable
)
{
}

nn_float
nn_get_input
(
	nn_net	*n,
	nn_uint	 index
)
{
	return 0.0;
}
