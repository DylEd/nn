#ifndef __NN_ERROR
#define __NN_ERROR

#include <stdio.h>
#include <stdlib.h>

void
error(	char	*mess,
	int	 val)
{
	printf("%s",mess);
	exit(1);
}

#endif /* __NN_ERROR */

