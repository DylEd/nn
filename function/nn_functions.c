#include "nn_functions.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define new_func malloc(sizeof(nn_activation_function))
#define new_loss_func malloc(sizeof(nn_loss_function))

#define new_coe(n) malloc((n) * sizeof(nn_float))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	ACTIVATION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn_activation_function *
nn_create_function
(
	nn_function_t	  f,
	nn_function_t	  f_prime,
	void		 *extra,
	void		(*extra_free)
			(
				void	*extra
			)
)
{
	return nn_create_function_with_coefficients(f,f_prime,0,0,extra,extra_free);
}

nn_activation_function *
nn_create_function_with_coefficients
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
)
{
	nn_activation_function *new_f = new_func;

	*new_f = (nn_activation_function) {
		.f			= f,
		.f_prime		= f_prime,
		.coefficients_count	= coefficients_count,
		.coefficients		= coefficients,
		.extra			= extra,
		.extra_free		= extra_free,
	};

	return new_f;
}

void
nn_free_function
(
	nn_activation_function	*function
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	SINGLE FOLD FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static
inline
nn_float
sum_arr
(
	nn_uint		n,
	nn_float	i[n]
)
{
	nn_float s = 0.0;
	for(int i_x; i_x < n; i_x++)
	{
		s += i[i_x];
	}
	return s;
}

nn_status
nn_activate_activation_function
(
	nn_activation_function	 *function,
	nn_uint			  inputs_count,
	nn_float		  inputs[inputs_count],
	nn_neuron		 *neuron,
	nn_uint			 *outputs_count,
	nn_float		**outputs
)
{
	function->f(
		inputs_count,
		inputs,
		function->coefficients_count,
		function->coefficients,
		neuron,
		function->extra,
		outputs_count,
		outputs
	);

	return 0;
}

nn_status
nn_activate_activation_function_prime
(
	nn_activation_function	 *function,
	nn_uint			  inputs_count,
	nn_float		  inputs[inputs_count],
	nn_neuron		 *neuron,
	nn_uint			 *outputs_count,
	nn_float		**outputs
)
{
	function->f_prime(
		inputs_count,
		inputs,
		function->coefficients_count,
		function->coefficients,
		neuron,
		function->extra,
		outputs_count,
		outputs
	);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//	IDENTITY
////////////////////////////////////////////////////////////////////////////////
void
identity
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = sum;
}

void
identity_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));
	
	memset(*outputs,0,inputs_count * sizeof(nn_float));
}

nn_activation_function *
nn_identity_function
(
)
{
	nn_activation_function *f = new_func;

	*f = (nn_activation_function) {
		.f		= identity,
		.f_prime	= identity_prime,
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	LINEAR
////////////////////////////////////////////////////////////////////////////////
void
linear
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	*outputs[0] = coefficients[0] * sum + coefficients[1];
}

void
linear_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = coefficients[0];
	}
}

nn_activation_function *
nn_linear_function
(
	nn_float	m,
	nn_float	b
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 2;

	nn_float *coe = new_coe(coe_num);

	coe[0] = m;
	coe[1] = b;

	*f = (nn_activation_function) {
		.f			= linear,
		.f_prime		= linear_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	BINSTEP
////////////////////////////////////////////////////////////////////////////////
void
binstep
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = sum < coefficients[0] ? 0.0 : 1.0;
}

void
binstep_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	memset(*outputs,0,inputs_count * sizeof(nn_float));
}

nn_activation_function *
nn_binstep_function
(
	nn_float	t
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = t;

	*f = (nn_activation_function) {
		.f			= binstep,
		.f_prime		= binstep_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	LOGISTIC
////////////////////////////////////////////////////////////////////////////////
void
logistic
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = 1/(1 + exp(coefficients[0] * -sum));
}

void
logistic_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	nn_float l = 1/(1 + exp(-sum));

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = coefficients[0] * l * (1-l);
	}
}

nn_activation_function *
nn_logistic_function
(
	nn_float	a
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;

	*f = (nn_activation_function) {
		.f			= logistic,
		.f_prime		= logistic_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	SMHT
////////////////////////////////////////////////////////////////////////////////
void
smht
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = (exp(coefficients[0] * sum) - exp(coefficients[1] * -sum)) / (exp(coefficients[2] * sum) + exp(coefficients[3] * -sum));
}

void
smht_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	nn_float ea = exp(coefficients[0] * sum);
	nn_float eb = exp(coefficients[1] * -sum);
	nn_float ec = exp(coefficients[2] * sum);
	nn_float ed = exp(coefficients[3] * -sum);

	nn_float y = (ea - eb) / (ec + ed);

	nn_float y_p = (coefficients[0] * ea - coefficients[1] * eb - y * (coefficients[2] * ec - coefficients[3]*ed)) / (ec + ed);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = y_p;
	}
}

nn_activation_function *
nn_smht_function
(
	nn_float	a,
	nn_float	b,
	nn_float	c,
	nn_float	d
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 4;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;
	coe[1] = b;
	coe[2] = c;
	coe[3] = d;

	*f = (nn_activation_function) {
		.f			= smht,
		.f_prime		= smht_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	TANH
////////////////////////////////////////////////////////////////////////////////
void
tanh_f
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = tanh(sum);
}

void
tanh_f_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] =  1 - pow(tanh(sum),2);
	}
}

nn_activation_function *
nn_tanh_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= tanh_f,
		.f_prime		= tanh_f_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	PRELU
////////////////////////////////////////////////////////////////////////////////
void
prelu
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] =  sum < 0 ? coefficients[0] * sum : coefficients[1] * sum;
}

void
prelu_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = sum < 0 ? coefficients[0] : coefficients[1];
	}
}

nn_activation_function *
nn_prelu_function
(
	nn_float	a,
	nn_float	b
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;
	coe[1] = b;

	*f = (nn_activation_function) {
		.f			= prelu,
		.f_prime		= prelu_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	LEAKY_RELU
////////////////////////////////////////////////////////////////////////////////
void
leaky_relu
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = sum < 0 ? 0.01 * sum : sum;
}

void
leaky_relu_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = sum < 0 ? 0.01 : 1;
	}
}

nn_activation_function *
nn_leaky_relu_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= leaky_relu,
		.f_prime		= leaky_relu_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	RELU
////////////////////////////////////////////////////////////////////////////////
void
relu
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = sum < 0 ? 0.0 : sum;
}

void
relu_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = sum < 0 ? 0.0 : 1;
	}
}

nn_activation_function *
nn_relu_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= relu,
		.f_prime		= relu_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	GELU
////////////////////////////////////////////////////////////////////////////////
void
gelu
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	(*outputs)[0] = (sum / 2) * (1 + erf(sum / sqrt(2)));
}

nn_float
gelu_phi
(
	nn_float	x
)
{
	return 0.5 * (1 + erf(x / sqrt(M_PI)));
}

void
gelu_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float sum = sum_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = gelu_phi(sum) + sum * gelu_phi(sum);
	}
}

nn_activation_function *
nn_gelu_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= gelu,
		.f_prime		= gelu_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	MULTI FOLD FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//	MAX_POOL
////////////////////////////////////////////////////////////////////////////////
static
inline
nn_float
max_arr
(
	nn_uint		xn,
	nn_float	x[xn]
)
{
	nn_float m = x[0];

	for(int i = 1; i < xn; i++)
	{
		if(m < x[i])
		{
			m = x[i];
		}
	}

	return m;
}

void
max_pool
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	(*outputs)[0] = max_arr(inputs_count,inputs);
}

void
max_pool_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float m = max_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = m == inputs[i] ? 1.0 : 0.0;
	}
}

nn_activation_function *
nn_max_pool_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= max_pool,
		.f_prime		= max_pool_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	MIN_POOL
////////////////////////////////////////////////////////////////////////////////
static
inline
nn_float
min_arr
(
	nn_uint		xn,
	nn_float	x[xn]
)
{
	nn_float m = x[0];

	for(int i = 1; i < xn; i++)
	{
		if(m > x[i])
		{
			m = x[i];
		}
	}

	return m;
}

void
min_pool
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	(*outputs)[0] = min_arr(inputs_count,inputs);
}

void
min_pool_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	nn_float m = min_arr(inputs_count,inputs);

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = m == inputs[i] ? 1.0 : 0.0;
	}
}

nn_activation_function *
nn_min_pool_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= min_pool,
		.f_prime		= min_pool_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}

////////////////////////////////////////////////////////////////////////////////
//	AVG_POOL
////////////////////////////////////////////////////////////////////////////////
static
inline
nn_float
avg_arr
(
	nn_uint		xn,
	nn_float	x[xn]
)
{
	nn_float sum = 0.0;

	for(int i = 0; i < xn; i++)
	{
		sum += x[i];
	}

	return sum / xn;
}

void
avg_pool
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	(*outputs)[0] = avg_arr(inputs_count,inputs);
}

void
avg_pool_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	nn_neuron	 *neuron,
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	for(int i = 0; i < inputs_count; i++)
	{
		(*outputs)[i] = 1.0 / inputs_count;
	}
}

nn_activation_function *
nn_avg_pool_function
(
)
{
	nn_activation_function *f = new_func;

	nn_uint coe_num = 0;

	nn_float *coe = new_coe(coe_num);

	*f = (nn_activation_function) {
		.f			= avg_pool,
		.f_prime		= avg_pool_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	LOSS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
nn_status
nn_activate_loss_function
(
	nn_loss_function	*function,
	nn_uint			 inputs_count,
	nn_float		 inputs[inputs_count],
	nn_float		 expected[inputs_count],
	nn_uint			 *outputs_count,
	nn_float		**outputs
)
{
	function->f(
		inputs_count,
		inputs,
		expected,
		function->coefficients_count,
		function->coefficients,
		function->extra,
		outputs_count,
		outputs
	);

	return 0;
}

nn_status
nn_activate_loss_function_prime
(
	nn_loss_function	*function,
	nn_uint			 inputs_count,
	nn_float		 inputs[inputs_count],
	nn_float		 expected[inputs_count],
	nn_uint			 *outputs_count,
	nn_float		**outputs
)
{
	function->f_prime(
		inputs_count,
		inputs,
		expected,
		function->coefficients_count,
		function->coefficients,
		function->extra,
		outputs_count,
		outputs
	);

	return 0;
}

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
	nn_loss_function *new_f = new_loss_func;

	*new_f = (nn_loss_function) {
		.f			= f,
		.f_prime		= f_prime,
		.coefficients_count	= coefficients_count,
		.coefficients		= coefficients,
		.extra			= extra,
		.extra_free		= extra_free,
	};

	return new_f;
}

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

////////////////////////////////////////////////////////////////////////////////
//	SOSE - SUM OF SQUARED ERRORS
////////////////////////////////////////////////////////////////////////////////
void
sose
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_float	  expected[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = 1;

	*outputs = realloc(*outputs,sizeof(nn_float));

	nn_float sum = 0.0;
	for(int i = 0; i < coefficients_count; i++)
	{
		sum += pow(inputs[i] - expected[i],2);
	}

	(*outputs)[0] = coefficients[0] * sum;
}

void
sose_prime
(
	nn_uint		  inputs_count,
	nn_float	  inputs[inputs_count],
	nn_float	  expected[inputs_count],
	nn_uint		  coefficients_count,
	nn_float	  coefficients[coefficients_count],
	void		 *extra,
	nn_uint		 *outputs_count,
	nn_float	**outputs
)
{
	*outputs_count = inputs_count;

	*outputs = realloc(*outputs,inputs_count * sizeof(nn_float));

	for(int i = 0; i < coefficients_count; i++)
	{
		(*outputs)[i] = coefficients[0] * 0.5 * fabs(inputs[i] - expected[i]);
	}
}

nn_loss_function *
nn_sose_loss_function
(
	nn_float	a
)
{
	nn_loss_function *f = new_loss_func;

	f = new_loss_func;

	nn_uint coe_num = 1;

	nn_float *coe = new_coe(coe_num);

	coe[0] = a;

	*f = (nn_loss_function) {
		.f			= sose,
		.f_prime		= sose_prime,
		.coefficients_count	= coe_num,
		.coefficients		= coe
	};

	return f;
}
