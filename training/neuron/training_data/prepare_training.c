#include <types.h>
#include <neuron_training_data.h>

#include <stdlib.h>
#include <string.h>

void
nn_neuron_prepare_training
(
	nn_neuron	*neuron,
	nn_uint		 recurrent_input_count
)
{
	nn_neuron_training_data *td = neuron->training_data;

	nn_uint rsize = recurrent_input_count * sizeof(nn_float *);

	#if defined NN_USE_NEURON_MULTI_OUTPUT
	td->outputs = malloc(rsize);
	#else
	td->outputs = malloc(recurrent_input_count * sizeof(nn_float));
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */
	td->deltas = malloc(rsize);
	td->from_outputs = malloc(rsize);

	for(int i = 0; i < recurrent_input_count; i++)
	{
		size_t fb = neuron->from_connections_count * sizeof(nn_float);
		td->deltas[i] = malloc(fb);
		memset(td->deltas[i],0,fb);
		td->from_outputs[i] = malloc(fb);
		memset(td->from_outputs[i],0,fb);

		#if defined NN_USE_NEURON_MULTI_OUTPUT
		size_t tb = neuron->to_connections_count * sizeof(nn_float);
		td->outputs[i] = malloc(tb);
		memset(td->outputs[i],0,tb);
		#endif /* NN_USE_NEURON_MULTI_OUTPUT */
	}
}
