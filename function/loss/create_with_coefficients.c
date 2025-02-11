#include <types.h>
#include <functions.h>
#include <loss_function.h>

#include "../function_helper.h"

#include <stdlib.h>

nn_loss_function *
nn_create_loss_function_with_coefficients
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
)
{
	nn_loss_function *lf = new_loss_func;

	*lf = (nn_loss_function) {
		.f			= f,
		.f_prime		= f_prime,
		.coefficients_count	= coefficients_count,
		.coefficients		= coefficients,
		.extra			= extra,
		.extra_free		= extra_free,
	};

	return lf;
}
