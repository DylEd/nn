#include <types.h>
#include <neuron_training_data.h>

#include <stdio.h>

void
nn_neuron_free_training_data
(
	nn_neuron_training_data	*training_data
)
{
	if(training_data->from_outputs	!= 0
		|| training_data->outputs != 0
		|| training_data->deltas != 0
	)
	{
		puts("training has not been finished");
		exit(1);
	}

	free(training_data);
}
