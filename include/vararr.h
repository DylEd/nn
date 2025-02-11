#ifndef __VARARR__
#define __VARARR__

////////////////////////////////////////////////////////////////////////////////
///	types
////////////////////////////////////////////////////////////////////////////////

/// standard variable for size
typedef unsigned int		 va_size_t;
/// standard variable for bool
typedef unsigned int		 va_bool_t;
/// standard variable for float
typedef double 			 va_float_t;
/// variable array type
typedef struct _va_t		*va_t;


////////////////////////////////////////////////////////////////////////////////
///	constructors
////////////////////////////////////////////////////////////////////////////////

///
va_t
va_new
(
	va_size_t	size
);
///
va_t
va_new_cap
(
	va_size_t	size,
	va_size_t	capacity
);
///
va_t
va_new_scl
(
	va_size_t	size,
	va_float_t	scale_factor
);
///
va_t
va_new_cap_scl
(
	va_size_t	size,
	va_size_t	capacity,
	va_float_t	scale_factor
);


////////////////////////////////////////////////////////////////////////////////
///	destructors
////////////////////////////////////////////////////////////////////////////////

///
void
va_free
(
	va_t	va
);


////////////////////////////////////////////////////////////////////////////////
///	accessors
////////////////////////////////////////////////////////////////////////////////

///
va_size_t
va_get_capacity
(
	va_t	va
);
///
va_size_t
va_get_length
(
	va_t	va
);
///
va_size_t
va_get_size
(
	va_t	va
);
///
va_float_t
va_get_scale
(
	va_t	va
);
///
void
va_set_scale
(
	va_t		va,
	va_size_t	scale_factor
);
///
void *
va_get_data
(
	va_t	va
);


////////////////////////////////////////////////////////////////////////////////
///	operators
////////////////////////////////////////////////////////////////////////////////

///
void
va_clear
(
	va_t	va
);
///
void
va_shift
(
	va_t	 va,
	void	*data
);
///
void
va_unshift
(
	va_t	 va,
	void	*storage
);
///
void
va_push
(
	va_t	 va,
	void	*data
);
///
void
va_pop
(
	va_t	 va,
	void	*storage
);
///
void
va_get
(
	va_t	 va,
	int	 index,
	void	*storage
);
///
void
va_set
(
	va_t	 va,
	int	 index,
	void	*data
);
///
void
va_remove
(
	va_t	 va,
	int	 index,
	void	*storage
);
///
void
va_insert
(
	va_t	 va,
	int	 index,
	void	*data
);
///
void
va_sort
(
	va_t	  va,
	int	(*compare)
		(
			const void *,
			const void *
		)
);
///
void
va_trim
(
	va_t	va
);

#endif // __VARARR__
