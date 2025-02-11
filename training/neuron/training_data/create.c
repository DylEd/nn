#include <types.h>
#include <neuron_training_data.h>

nn_neuron_training_data *
nn_neuron_create_training_data
(
	void
)
{
	nn_neuron_training_data *td = malloc(sizeof(nn_neuron_training_data));

	*td = (nn_neuron_training_data) {0};

	return td;
}
