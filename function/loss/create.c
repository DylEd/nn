#include <types.h>
#include <functions.h>
#include <loss_function.h>

#include <stdlib.h>

nn_loss_function *
nn_create_loss_function
(
	nn_loss_function_t	  f,
	nn_loss_function_t	  f_prime,
	void			 *extra,
	void			(*extra_free)
				(
					void	*extra
				)
)
{
	return nn_create_loss_function_with_coefficients(
		f,
		f_prime,
		0,
		0,
		extra,
		extra_free
	);
}
