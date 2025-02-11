#ifndef __NETWORK_TRAINING_DATA
#define __NETWORK_TRAINING_DATA

#include <stdio.h>
#include <stdlib.h>
#include <types.h>
//#include <nn_error.h>

////////////////////////////////////////////////////////////////////////////////
//	NET TRAINING DATA
////////////////////////////////////////////////////////////////////////////////
struct nn_network_training_data
{
	nn_loss_function	*loss_function;

	nn_uint			 r_input_count;		// total number of inputs for current set of inputs (recurrence)
	nn_uint			 r_input_num;		// current input in input_num -th set of inputs

	// provided by user ~ nn_model > nn_net_train > parameter[input]
	nn_float		*loss_deltas;		// [m->output_neurons_count]
	nn_float		*outputs;		// [m->output_neurons_count]
	nn_float		*expected_outputs;	// [input_count] ~ from nn_model > nn_net_train > parameter
};

nn_network_training_data *
nn_network_training_data_create
(
	void
);

void
nn_network_training_data_free
(
	nn_network_training_data	*training_data
);

#endif /* __NETWORK_TRAINING_DATA */
