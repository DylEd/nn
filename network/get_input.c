#include <types.h>
#include <network.h>

nn_float
nn_net_get_input
(
	nn_net	*network,
	nn_uint	 index
)
{
	return network->network_input[index];
}
