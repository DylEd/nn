#ifndef __NN_FUNCTIONS
#define __NN_FUNCTIONS

#include <types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	ACTIVATION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn_activation_function *
nn_identity_function
(
	void
);

nn_activation_function *
nn_linear_function
(
	nn_float	m,
	nn_float	b
);

nn_activation_function *
nn_binstep_function
(
	nn_float t
);

nn_activation_function *
nn_logistic_function
(
	nn_float	a
);

nn_activation_function *
nn_smht_function
(
	nn_float	a,
	nn_float	b,
	nn_float	c,
	nn_float	d
);

nn_activation_function *
nn_tanh_function
(
	void
);

nn_activation_function *
nn_prelu_function
(
	nn_float	a,
	nn_float	b
);

nn_activation_function *
nn_leaky_relu_function
(
	void
);

nn_activation_function *
nn_relu_function
(
	void
);

nn_activation_function *
nn_gelu_function
(
	void
);

nn_activation_function *
nn_max_pool_function
(
	void
);

nn_activation_function *
nn_min_pool_function
(
	void
);

nn_activation_function *
nn_avg_pool_function
(
	void
);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	LOSS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//sum of squared error
nn_loss_function *
nn_sose_loss_function
(
	nn_float	a
);

#endif /* __NN_FUNCTIONS */
