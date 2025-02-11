#ifndef __NETWORK
#define __NETWORK

#include <types.h>

/*
	create a network
*/
nn_net *
nn_net_create
(
	void
);

/*
	free network

	\param network	network to free
*/
void
nn_net_free
(
	nn_net	*network
);

/*
	get the index-th input to the network

	\param network	network containing input
	\param index	index of input to retrieve
*/
nn_float
nn_net_get_input
(
	nn_net	*network,
	nn_uint	index
);

/*
	get outputs

	\param network
	\return
*/
nn_float *
nn_net_get_output
(
	nn_net	*network
);

/*
	load set of inputs and run network

	\param network	network to run
	\param input	set of inputs
*/
void
nn_net_run
(
	nn_net		*network,
	nn_float	*input
);

void
nn_net_train_backpropagation
(
	nn_net			 *m,
	nn_uint			  inputs_count,
	nn_float		 *inputs[inputs_count],			// [m->input_neurons_count]
	nn_uint			  r_inputs[inputs_count],
	nn_float		 *expected_outputs[inputs_count],	// [m->output_neurons_count]
	nn_uint			  epochs,
	nn_float		  training_rate,
	nn_loss_function	  loss_function,
	nn_uint			(*callback_function)
				(
					nn_net		*m,
					nn_float	 loss,
					nn_uint		 epoch
				)
);

/*
	arrange neurons and set neuron->order

	\param network	network to be ordered
*/
void
nn_net_order_neurons
(
	nn_net	*network
);

/*
	add a network to another network

	All neurons of sub_net are cloned and added to network.
	NN_OUTPUT_NEURON 's and NN_INPUT_NEURON 's are changed to 
	NN_HIDDEN_NEURON 's. If immutable is true, then the connections
	within sub_net will be made immutable; otherwise, they are left
	unchanged. The neurons of to_inputs and from_outputs will be fully
	connected to the inputs and outputs of sub_net, respectively.

	\param network			network to which sub_net is added
	\param sub_net			network which is added to net
	\param to_inputs_count		size of to_inputs
	\param to_inputs		neurons of network which connect to the
						input neurons of sub_net
	\param from_outputs_count	size of from_outputs
	\param from_outputs		neurons of network which connect from the
						output neurons of sub_net
	\param immutable		whether the weights within sub_net should change
*/
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
);

/*
	add a neuron to a network

	\param net	network to which neuron is added
	\param neuron	neuron which is added
*/
void
nn_net_push_neuron
(
	nn_net		*net,
	nn_neuron	*neuron
);

#endif /* __NETWORK */
