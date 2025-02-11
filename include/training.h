#ifndef __NN_TRAINING_DATA
#define __NN_TRAINING_DATA

#include <net_training_data.h>
#include <neuron_training_data.h>

/*
	activate the neuron and store data used for training

	\param net	net containing the neuron
	\param neuron	neuron to activate
*/
void
nn_neuron_activate_training
(
	nn_net		*net,
	nn_neuron	*neuron
);

/*
	backpropagate through the neuron

	behavior by neuron_type
	NN_INPUT_NEURON:
		do nothing
	NN_HIDDEN_NEURON:
		typical backpropagation
	NN_OUTPUT_NEURON:
		if r_input_num == r_input_count (the neuron is output for the final unwrap)
			retrieve incoming delta from net; otherwise, typical backpropagation
		otherwise
			incoming delta and outgoing deltas are 0
		-- this may change in the future to allow intermediate error adjustments
	NN_RECURRENT_NEURON:
		if r_input_num == r_input_count (in the final unwrap)
			incoming delta and outgoing deltas are 0

	\param net	network containing the neuron
	\param neuron	neuron to backpropagate
*/
void
nn_neuron_backprop_training
(
	nn_net		*net,
	nn_neuron	*neuron
);

#endif /* __NN_TRAINING_DATA */

