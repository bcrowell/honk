#define HONK_ERR_UNDEFINED_FN 1
#define HONK_ERR_SPLINE_TOO_LARGE 2
#define HONK_ERR_TOO_MANY_PARTIALS 3
#define HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE 4
#define HONK_ERR_NAN 5
#define HONK_INDEX_OUT_OF_RANGE 6
#define HONK_ILLEGAL_VALUE 7

// When changing one of the following, need to do a "make cpu_c.so".

#define MAX_INSTANCES 65536
/* ... used to set size of error return array; this is in global memory, which is plentiful, so not much harm in making it much bigger than the
       number of instances ewe actually expect to have
*/
#define MAX_SPLINE_KNOTS 300
#define A_SPLINE_ORDER 3
#define PHASE_SPLINE_ORDER 4
#define MAX_SPLINE_COEFFS (MAX_SPLINE_KNOTS*(PHASE_SPLINE_ORDER+1))
// ... total number of cubic spline coefficients in all partials
#define MAX_PARTIALS 16
