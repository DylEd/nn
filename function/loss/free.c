#include <types.h>
#include <functions.h>

#include <stdlib.h>

void
nn_free_loss_function
(
	nn_loss_function	*function
)
{
	if(function->coefficients != 0)
	{
		free(function->coefficients);
	}
	if(function->extra != 0)
	{
		if(function->extra_free != 0)
		{
			function->extra_free(function->extra);
		}
		else
		{
			free(function->extra);
		}
	}

	free(function);
}
