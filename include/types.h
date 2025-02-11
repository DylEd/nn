#ifndef __NN_TYPES
#define __NN_TYPES

#include <stdarg.h>

typedef struct nn_neuron nn_neuron;
typedef struct nn_layer nn_layer;
typedef struct nn_net nn_net;

typedef struct nn_connection nn_connection;
typedef struct nn_activation_function nn_activation_function;
typedef struct nn_loss_function nn_loss_function;

typedef struct nn_neuron_training_data nn_neuron_training_data;
typedef struct nn_network_training_data nn_network_training_data;

typedef double nn_float;
typedef unsigned int nn_uint;
typedef bool nn_bool;
typedef int nn_status;

typedef void (*nn_function_t)
	(
		nn_uint		  inputs_count,
		nn_float	  inputs[inputs_count],
		nn_uint		  coefficients_count,
		nn_float	  coefficients[coefficients_count],
		nn_neuron	 *neuron,
		void		 *extra,
		nn_uint		 *outputs_count,
		nn_float	**outputs
	);

typedef void (*nn_loss_function_t)
	(
		nn_uint		  inputs_count,
		nn_float	  inputs[inputs_count],
		nn_float	  expected[inputs_count],
		nn_uint		  coefficients_count,
		nn_float	  coefficients[coefficients_count],
		void		 *extra,
		nn_uint		 *outputs_count,
		nn_float	**outputs
	);

typedef
enum
{
	NN_HIDDEN_NEURON = 0,
	NN_INPUT_NEURON,
	NN_OUTPUT_NEURON,
	NN_RECURRENT_NEURON,
} nn_neuron_type;

typedef
enum
{
	NN_INPUT_LAYER,
	NN_HIDDEN_LAYER,
	NN_OUTPUT_LAYER,
	NN_RECURRENT_LAYER,
} nn_layer_type;

enum
{
	NN_NEURON,
	NN_NET,
};

typedef
enum
{
	NN_REPEAT,
	NN_REPEAT_REVERSE,
	NN_REPEAT_LAST_ELEMENT,
	NN_REPEAT_SNAKE,
} nn_repeat_type;

enum
{
	NN_SUCCESS = 0,
	NN_FAIL,
};

////////////////////////////////////////////////////////////////////////////////
//	CONNECTION
////////////////////////////////////////////////////////////////////////////////
struct nn_connection
{
	nn_neuron	*from_neuron;
	nn_neuron	*to_neuron;
	nn_float	 weight;
	nn_bool		 immutable;
};

////////////////////////////////////////////////////////////////////////////////
//	ACTIVATION FUNCTION
////////////////////////////////////////////////////////////////////////////////
struct nn_activation_function
{
	nn_function_t	  f;
	nn_function_t	  f_prime;

	nn_float	 *coefficients;
	nn_uint		  coefficients_count;

	void		 *extra;
	void		(*extra_free)
			(
				void	*extra
			);
};

////////////////////////////////////////////////////////////////////////////////
//	LOSS FUNCTION
////////////////////////////////////////////////////////////////////////////////
struct nn_loss_function
{
	nn_loss_function_t		  f;
	nn_loss_function_t		  f_prime;

	nn_float			 *coefficients;
	nn_uint				  coefficients_count;

	void				 *extra;
	void				(*extra_free)
					(
						void	*extra
					);
};

////////////////////////////////////////////////////////////////////////////////
//	NEURON
////////////////////////////////////////////////////////////////////////////////
struct nn_neuron
{
	nn_connection		**from_connections;
	nn_uint			  from_connections_count;

	nn_connection		**to_connections;
	nn_uint			  to_connections_count;

	nn_activation_function	 *activation_function;

	nn_uint			  neuron_type;

	#if defined NN_USE_NEURON_MUTLI_OUTPUT
	nn_float		 *output;			// [to_connections_count]
	//nn_uint			  repeat_type;			// what outputs to send to next neuron when activation function
	#else // use single output
	nn_float		  output;
	#endif /* NN_USE_NEURON_MULTI_OUTPUT */

	const nn_uint		  id;
	nn_uint			  order;

	nn_neuron_training_data			 *training_data;

	void			 *extra;
};

////////////////////////////////////////////////////////////////////////////////
//	NETWORK
////////////////////////////////////////////////////////////////////////////////
struct nn_net
{
	// architecture
	nn_neuron		**neurons;
	nn_uint			  neurons_count;

	nn_neuron		**hidden_neurons;		// points into neurons
	nn_uint			  hidden_neurons_count;

	nn_neuron		**input_neurons;		// points into neurons
	nn_uint			  input_neurons_count;

	nn_neuron		**output_neurons;		// points into neurons
	nn_uint			  output_neurons_count;

	nn_neuron		**recurrent_neurons;		// points into neurons
	nn_uint			  recurrent_neurons_count;

	// running
	nn_float		 *network_input;		// [input_neurons_count]

	// training
	//  only available during training
	nn_network_training_data			 *training_data;
};

#endif /* __NN_TYPES */
