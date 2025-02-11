#ifndef __ACTIVATION_FUNCTION
#define __ACTIVATION_FUNCTION

#include <types.h>

nn_activation_function *
nn_activation_function_create
(
	nn_function_t	  f,
	nn_function_t	  f_prime,
	void		 *extra,
	void		(*extra_free)
			(
				void	*extra
			)
);

nn_activation_function *
nn_activation_function_create_with_coefficients
(
	nn_function_t	  f,
	nn_function_t	  f_prime,
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	void		 *extra,
	void		(*extra_free)
			(
				void	*extra
			)
);

void
nn_activation_function_free
(
	nn_activation_function *function
);

nn_status
nn_activation_function_activate
(
	nn_neuron	*neuron,
	nn_float	*inputs,	// [neuron->from_connections_count]
	nn_float	*outputs	// [neuron->to_connections_count]
);

nn_status
nn_activation_function_prime_activate
(
	nn_neuron	*neuron,
	nn_float	*inputs,	// [neuron->from_connections_count]
	nn_float	*outputs	// [neuron->to_connections_count]
);

/*
nn_status
nn_activation_function_activate
(
	nn_activation_function	 *function,
	nn_uint			  inputs_count,
	nn_float		  inputs[inputs_count],
	nn_neuron		 *neuron,
	nn_uint			 *outputs_count,
	nn_float		**outputs
);

nn_status
nn_activation_function_prime_activate
(
	nn_activation_function	 *function,
	nn_uint			  inputs_count,
	nn_float		  inputs[inputs_count],
	nn_neuron		 *neuron,
	nn_uint			 *outputs_count,
	nn_float		**outputs
);
*/

#endif /* __ACTIVATION_FUNCTION */
