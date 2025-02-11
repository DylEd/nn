#ifndef __LOSS_FUNCTION
#define __LOSS_FUNCTION

#include <types.h>

nn_loss_function *
nn_loss_function_create
(
	nn_loss_function_t	  f,
	nn_loss_function_t	  f_prime,
	void			 *extra,
	void			(*extra_free)
				(
					void	*extra
				)
);

nn_loss_function *
nn_loss_function_create_with_coefficients
(
	nn_loss_function_t	  f,
	nn_loss_function_t	  f_prime,
	nn_uint			  coefficients_count,
	nn_float		  coefficients[coefficients_count],
	void			 *extra,
	void			(*extra_free)
				(
					void	*extra
				)
);

void
nn_loss_function_free
(
	nn_loss_function	*function
);

nn_status
nn_loss_function_activate
(
	nn_loss_function	*function,
	nn_uint			 input_counts,
	nn_float		 actual[input_counts],
	nn_float		 expected[input_counts],
	nn_uint			 *outputs_count,
	nn_float		**outputs
);

nn_status
nn_loss_function_prime_activate
(
	nn_loss_function	*function,
	nn_uint			 input_counts,
	nn_float		 actual[input_counts],
	nn_float		 expected[input_counts],
	nn_uint			 *outputs_count,
	nn_float		**outputs
);

#endif /* __LOSS_FUNCTION */
