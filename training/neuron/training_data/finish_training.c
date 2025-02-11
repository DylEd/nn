#include <types.h>
#include <neuron_training_data.h>

void
nn_neuron_finish_training
(
	nn_net		*net,
	nn_neuron	*neuron
)
{
	nn_neuron_training_data *training_data = neuron->training_data;

	if(training_data->from_outputs	== 0
		|| training_data->outputs == 0
		|| training_data->deltas == 0
	)
	{
		puts("training has not been started");
		exit(1);
	}

	for(int i = 0; i < net->training_data->r_input_count; i++)
	{
		#if defined NN_USE_NEURON_MULTI_OUTPUT
		if(training_data->outputs[i] != 0)
		{
			free(training_data->outputs[i]);
		}
		#endif /* NN_USE_NEURON_MULTI_OUTPUT */
		if(training_data->deltas[i] != 0)
		{
			free(training_data->deltas[i]);
		}
		if(training_data->from_outputs[i] != 0)
		{
			free(training_data->from_outputs[i]);
		}
	}

	free(training_data->outputs);
	free(training_data->deltas);
	free(training_data->from_outputs);

	training_data->outputs = 0;
	training_data->deltas = 0;
	training_data->from_outputs = 0;

	nn_neuron_free_training_data(training_data);

	neuron->training_data = 0;
}
