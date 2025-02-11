#ifndef __NEURON
#define __NEURON

#include <types.h>

nn_activation_function *nn_default_activation_function;

/*
	create a neuron on an nn_network net configured as specified by neuron_spec
	- neuron_spec should only define the following, anything else is overwritten:
		-- always available
			--- activation_function	(default: nn_default_activation_function)
			--- neuron_type		(default: NN_HIDDEN_NEURON)
			--- extra		(default: 0)
		-- only available when NN_USE_NEURON_MUTLI_OUTPUT is defined
			--- repeat_type		(default: )

	macro-shorthand: nn_neuron_create(net,.field1 = a,.field2 = b)

	\param net		network in which the new neuron is created
	\param neuron_spec	configuration of the new neuron
*/
nn_neuron *
nn_neuron_create_from_struct
(
	nn_net			*net,
	nn_neuron		 neuron_spec
);
#define nn_neuron_create(net,...) nn_neuron_create_from_struct(net,(nn_neuron){...})

/*
	free the neuron

	\param net	net containing the neuron
	\param neuron	neuron to be freed
*/
void
nn_neuron_free
(
	nn_net		*net,
	nn_neuron	*neuron
);

/*
	activate the neuron

	\param net	net containing the neuron
	\param neuron	neuron to activate
*/
void
nn_neuron_activate
(
	nn_net		*net,
	nn_neuron	*neuron
);

/*
	clone the neuron

	\param neuron	neuron to clone
	\return		cloned neuron
*/
nn_neuron *
nn_neuron_clone
(
	nn_neuron	*neuron
);

#endif /* __NEURON */
