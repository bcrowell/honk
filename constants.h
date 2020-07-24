#define HONK_FN_OSC 1
#define HONK_FN_ZETA 2

#define HONK_ERR_UNDEFINED_FN 1
#define HONK_ERR_SPLINE_TOO_LARGE 2
#define HONK_ERR_TOO_MANY_PARTIALS 3
#define HONK_ERR_TOO_MANY_KNOTS_IN_SPLINE 4

#define MAX_SPLINE_KNOTS 80
#define SPLINE_ORDER 3
#define MAX_SPLINE_COEFFS (MAX_SPLINE_KNOTS*(SPLINE_ORDER+1))
// ... total number of cubic spline coefficients in all partials
#define MAX_PARTIALS 16
