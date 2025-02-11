#ifndef __CONNECTION
#define __CONNECTION

#include <types.h>

nn_connection *
nn_connection_create
(
	nn_neuron	*from_neuron,
	nn_neuron	*to_neuron,
	nn_float	 weight,
	nn_bool		 immutable
);

nn_connection *
nn_connection_connect
(
	nn_neuron	*from_neuron,
	nn_neuron	*to_neuron,
	nn_float	 weight,
	nn_bool		 immutable
);

void
nn_connection_immutable
(
	nn_connection	*connection
);

void
nn_connection_mutable
(
	nn_connection	*connection
);

nn_connection *
nn_connection_find
(
	nn_neuron	*from_neuron,
	nn_neuron	*to_neuron
);

void
nn_connection_update_weight
(
	nn_connection	*connection,
	nn_float	 weight
);

void
nn_conection_free
(
	nn_connection	*connection
);

void
nn_connection_free_all
(
	nn_uint		 n,
	nn_connection	*connection[n]
);

nn_float
(*nn_connection_default_weight)
(
	void
);

#endif /* __CONNECTION */
