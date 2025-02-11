#ifndef __NEURON_TRAINING_DATA
#define __NEURON_TRAINING_DATA

#include <types.h>
//#include <nn_error.h>

#include <stdio.h>
#include <stdlib.h>

#include <net_training_data.h>

////////////////////////////////////////////////////////////////////////////////
//	NEURON DATA
////////////////////////////////////////////////////////////////////////////////

// allocated by nn_neuron_init_training
// freed by nn_net_end_training
struct nn_neuron_training_data
{
	nn_uint		  type_order;		// order within net->{neuron_type}_neurons

	nn_float	**from_outputs;		// list of lists of from connection outputs	[m->r_input_count][n->from_connections_count]
	#if defined NN_USE_NEURON_MUTLI_OUTPUT
	nn_float	**outputs;		// list of outputs				[m->r_input_count][n->to_connections_count]
	#else
	nn_float	 *outputs;		// list of outputs				[m->r_input_count]
	#endif	/* NN_USE_NEURON_MUTLI_OUTPUT */
	nn_float	**deltas;		// list of deltas w.r.t. from connection	[m->r_input_count][n->from_connections_count]
};

/*
	create training data

	prefer use of nn_net_init_training
*/
nn_neuron_training_data *
nn_neuron_create_training_data
(
	void
);

/*
	free training data

	prefer use of nn_net_finish_training

	otherwise, must first call nn_neuron_finish_training
		or have the following:
			training_data->outputs == 0
			training_data->deltas == 0
			training_data->from_outputs == 0
*/
void
nn_neuron_free_training_data
(
	nn_neuron_training_data	*training_data
);

/*
	free the fields of neuron->training_data

	\param net	net containing neuron
	\param neuron	neuron to finish training
*/
void
nn_neuron_finish_training
(
	nn_net		*net,
	nn_neuron	*neuron
);

/*
	allocate and initialize the fields of neuron->training_data

	\param neuron			neuron to prepare
	\param recurrent_input_count	number of recurrent in cycle
*/
void
nn_neuron_prepare_training
(
	nn_neuron	*neuron,
	nn_uint		 recurrent_input_count
);

#endif /* __NEURON_TRAINING_DATA */
