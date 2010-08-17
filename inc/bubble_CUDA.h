#ifndef _BUBBLECUDA_H_
#define _BUBBLECUDA_H_

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <assert.h>
#include <complex.h>
#include <cstddef>
#include <vector_types.h>
#include <math.h>
#include <float.h>

#include <climits>

#include <cuda_runtime.h>
#include <cutil_inline.h>
//#include <GL/glew.h>
//#include <GL/glut.h>
//#include <cuda_gl_interop.h>
//#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
//#include <rendercheck_gl.h>


#include <cutil_math.h>
#include <cutil.h>
//#include <cudpp.h>

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/extrema.h"
#include "thrust/reduce.h"
#include "thrust/functional.h"
#include "thrust/sort.h"
#include "thrust/is_sorted.h"
#include "thrust/gather.h"
#include "thrust/scatter.h"

#define TRUE 1
#define FALSE 0

//#define _DEBUG_
#define _OUTPUT_

#define EPSILON 0.01

#define DIMENSION 2

#define TILE_BLOCK_WIDTH	16
#define TILE_BLOCK_HEIGHT	16
#define LINEAR_BLOCK_SIZE	256

struct __align__(16) debug_t
{
    int display;	// Number of Lines to display during diagnostics
    // Switches for display
    bool fg;	// Void Fraction
    bool p0;	// Pressure
    bool T;		// Temperature
    bool vxy;	// Velocity
    bool bubbles;	// Bubbles
};

struct __align__(16) array_index_t
{
//	int	lmax;
    int	ms, me, ns, ne;			// Offsets for midpoint, nodepoint
    int	istam, iendm, istan, iendn;
    int	ista1m, iend1m, ista1n, iend1n;
    int	ista2m, iend2m, ista2n, iend2n;
    int	jstam, jendm, jstan, jendn;
    int	jsta1m, jend1m, jsta1n, jend1n;
    int	jsta2m, jend2m, jsta2n, jend2n;
};

struct __align__(8) grid_t
{
    int	X, Y;		// Grid dimension in cells
    double	LX, LY;		// Grid dimension in millimeters
    double	dx, dy;		// Cell dimension in millimeters
    double	rdx, rdy;	// Radial dimensions in millimeters
};

struct __align__(16) grid_gen
{
    double	*xu, *rxp;	// array pointers for xu and rxp
    int xu_size, rxp_size;
};

struct __align__(16) sigma_t
{
    double *mx, *my;	// Sigma for midpoints
    double *nx, *ny;	// Sigma for nodepoints
    int mx_size, my_size;
    int nx_size, ny_size;
};

struct __align__(16) PML_t
{
    bool	X0, X1, Y0, Y1;	// Flags for PML boundaries
    int	NPML;		// Depth of PML in cells
    int	order;		// PML order
    double	sigma;		// sigma used for PML layer
};

struct __align__(16) plane_wave_t
{
    double	amp;		// Wave amplitude
    double	freq;		// Frequency
    double	f_dist;		// Focal distance
    double	box_size;	// Size of the "box"
    bool	cylindrical;	// Flag for cylindrical condition
    int	on_wave;
    int	off_wave;
    double2	fp;		// Focal point (calculated from focal distance)
    double	omega;
    bool	Plane_P, Plane_V, Focused_P, Focused_V;	// Flags for types of waves
};

struct __align__(16) sim_params_t
{
    double	cfl;		// Courant number
    double	dt0;		// Timestep size
    int	order;		// Order of FDTD
    int	deltaBand;	// Bandwidth of smooth delta function
    unsigned int	NSTEPMAX;	// Maximum number of steps in simulation
    double	TSTEPMAX;		// Maximum simulation time
    int	DATA_SAVE;	// Interval to save images
};

struct __align__(16) bub_params_t
{
    bool	enabled;	// Switch for enabling bubbles
    double	fg0;		// Initial void fraction
    double	R0, R03;	// Initial radius. We cache R0^3 as well to save ops
    double	PL0;		// Initial liquid pressure
    double	PG0;		// Initial gas pressure
    double	T0;		// Initial temperature
    double	gam;		// Gamma
    double	sig;		// Surface tension
    double	mu;		// Viscosity
    double	rho;		// Density
    double	K0;		// Heat capacity
    double	coeff_alpha;	// Coefficient used in calculation of Alpha_N
    int	mbs, mbe;
    int	nbs, nbe;
    int	np, npi;	// Number of bubbles
    double	dt0;		// Bubble default timestep
};

struct __align__(16) mix_params_t
{
    double	T_inf;
    double	P_inf;
    double	fg_inf;
    double	rho_inf;
    double	cs_inf;
    double	dt;		// Mixture timestep
};
struct __align__(16) mixture_t
{
    double	*T;			// Temperature
    double	*p0;			// Magnitude of pressure
    double2	*p, *pn;		// Pressure components
    double	*vx, *vy;		// Velocity
    double	*c_sl;			// Speed of sound
    double	*rho_m, *rho_l;		// Density
    double	*f_g,*f_gn,*f_gm;	// Void fraction
    double	*k_m, *C_pm;		// Km and Heat capacity
    double	*Work;			// Temporary array
    double	*Ex, *Ey;		// Mixture energy
};

struct __align__(16) bubble_t
{
    int2	*ibm;		// index in mid cell
    int2	*ibn;		// index in node cell
    double2	*pos;		// position
    double	*R_t;		// radius at mixture timestep
    double	*R_p, *R_pn;		// Radius variables used in Rayleigh Plesset solution
    double	*R_n, *R_nn;		//
    double	*d1_R_p, *d1_R_n;	// time derivative of radius
    double	*PG_p, *PG_n;		// gas pressure
    double	*PL_p, *PL_n, *PL_m;	// liquid pressure
    double	*Q_B;		// Bubble heat
    double	*n_B;		// number weight of reference bubble
    double	*dt, *dt_n;	// time increment
    double	*re, *re_n;	// remaining time
    double2	*v_B;	// velocity of bubble
    double2	*v_L;	// velocity of liquid around bubble
};

struct __align__(16) bubble_t_aos
{
    int2	ibm;		// index in mid cell
    int2	ibn;		// index in node cell
    double2	pos;		// position
    double	R_t;		// radius at mixture timestep
    double	R_p, R_pn;		// Radius variables used in Rayleigh Plesset solution
    double	R_n, R_nn;		//
    double	d1_R_p, d1_R_n;	// time derivative of radius
    double	PG_p, PG_n;		// gas pressure
    double	PL_p, PL_n, PL_m;	// liquid pressure
    double	Q_B;		// Bubble heat
    double	n_B;		// number weight of reference bubble
    double	dt, dt_n;	// time increment
    double	re, re_n;	// remaining time
    double2	v_B;	// velocity of bubble
    double2	v_L;	// velocity of liquid around bubble
};

struct output_plan_t
{
    mixture_t mixture_h;
    bubble_t bubbles_h;
    int step;
    array_index_t *array_index;
    grid_t *grid_size;
    sim_params_t *sim_params;
    plane_wave_t *plane_wave;
    debug_t *debug;
};

void display(double data[], int xdim, int ydim, int num_lines, char **msg);

    int solve_bubbles(array_index_t *array_index, grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, bub_params_t *bub_params, plane_wave_t *plane_wave, debug_t *debug, int argc, char ** argv);

    int initialize_variables (grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, plane_wave_t *plane_wave, array_index_t *array_index, mix_params_t *mix_params, bub_params_t *bub_params);

    int initialize_CUDA_variables(grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, plane_wave_t *plane_wave, array_index_t *array_index, mix_params_t *mix_params, bub_params_t *bub_params);
    int destroy_CUDA_variables(bub_params_t *bub_params);

    grid_t init_grid_size(grid_t grid_size);

    plane_wave_t init_plane_wave(plane_wave_t plane_wave, grid_t grid_size);

    array_index_t init_array (const grid_t grid_size, const sim_params_t sim_params);

    bub_params_t init_bub_params (bub_params_t bub_params, sim_params_t sim_params, double dt);

    grid_gen init_grid_vector(array_index_t array_index, grid_t grid_size);

    sigma_t init_sigma(const PML_t PML, const sim_params_t sim_params, const grid_t grid_size, const array_index_t array_index);
    mix_params_t init_mix();

    double mix_set_time_increment(sim_params_t sim_params, double dx_min, double u_max);
    mixture_t init_mix_array(mix_params_t *mix_params, array_index_t array_index);

    bubble_t init_bub_array(bub_params_t *bub_params, mix_params_t *mix_params, array_index_t *array_index, grid_t *grid_size, plane_wave_t *plane_wave);
    bubble_t_aos bubble_input(double2 pos, double fg_in, bub_params_t bub_params, grid_t grid_size, plane_wave_t plane_wave);

    void setCUDAflags();
    void checkCUDAError(const char* msg);
#endif


