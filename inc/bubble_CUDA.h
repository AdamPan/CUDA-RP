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
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>
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

#define _CYLINDRICAL_
#define _DEBUG_

#define EPSILON 0.01

#define DIMENSION 2

#define TILE_BLOCK_WIDTH	32
#define TILE_BLOCK_HEIGHT	8
//#define TILE_BLOCK_SIZE	16
#define LINEAR_BLOCK_SIZE	128
#define HALF_LINEAR_SIZE	64

#define VFRL_MAX_AREA	4
#define BH_MAX_AREA	4

#define WINDOW_HEIGHT	600
#define WINDOW_WIDTH	800


struct __align__(16) debug_t
{
	int display;	// Number of Lines to display during diagnostics
	// Switches for display
	bool fg;	// Void Fraction
	bool p0;	// Pressure
	bool pxy;	// Pressure (x and y components) 
	bool T;		// Temperature
	bool vxy;	// Velocity
	bool bubbles;
};

struct __align__(16) array_index_t
{
//	int	lmax;
	int	ms, me, ns, ne;
	int	istam, iendm, istan, iendn;
	int	ista1m, iend1m, ista1n, iend1n;
	int	ista2m, iend2m, ista2n, iend2n;
	int	jstam, jendm, jstan, jendn;
	int	jsta1m, jend1m, jsta1n, jend1n;
	int	jsta2m, jend2m, jsta2n, jend2n;
};

struct __align__(8) grid_t
{
	int	X, Y;
	double	LX, LY;
	double	dx, dy;
	double	rdx, rdy;
};

struct __align__(16) grid_gen
{
	double	*xu, *rxp;
	int xu_size, rxp_size;
};

struct __align__(16) sigma_t
{
	double *mx;
	double *my;
	double *nx;
	double *ny;
	int mx_size;
	int my_size;
	int nx_size;
	int ny_size;
};

struct __align__(16) PML_t
{
	bool	X0,X1,Y0,Y1;
	int	NPML;
	int	order;
	double	sigma;
};

struct __align__(16) plane_wave_t
{
	double	amp;
	double	freq;
	double	f_dist;
	double	box_size;
	bool	cylindrical;
	int	on_wave;
	int	off_wave;
	double2	fp;
	double	omega;
	bool	Plane_P, Plane_V, Focused_P, Focused_V;
};

struct __align__(16) sim_params_t
{
	double	cfl;
	double	dt0;
	int	order;
	int	deltaBand;
	
	unsigned int	NSTEPMAX;
	int	TSTEPMAX;
	int	DATA_SAVE;
	int	WRITE_20;
};

struct __align__(16) bub_params_t
{
	double	fg0;
	double	R0, R03;	// We cache R0^3 as well to save ops
	double	PL0;
	double	PG0;
	double	T0;
	double	gam;
	double	sig;
	double	mu;
	double	rho;
	double	K0;
	double	coeff_alpha;
	int	mbs, mbe;
	int	nbs, nbe;
	int	np, npi;
	double	dt0;
};

struct __align__(16) mix_params_t
{
	double	T_inf;
	double	P_inf;
	double	fg_inf;
	double	rho_inf;
	double	cs_inf;
	double	dt;
};
struct __align__(16) mixture_t
{
	// temperature
	double	*T;
	// pressure
	double	*p0;
	double2	*p, *pn;
	double	*vx, *vy;
	double	*c_sl;
	double	*rho_m, *rho_l;
	double	*f_g,*f_gn,*f_gm;
	double	*k_m, *C_pm;
	double	*Work;
	double	*Ex, *Ey;
};

struct __align__(16) bubble_t
{
	// index in mid cell
	int2	*ibm;
	// index in node cell
	int2	*ibn;
	// position
	double2	*pos;
	// radius
	double	*R_t;
	double	*R_p, *R_pn;
	double	*R_n, *R_nn;
	// time derivative of radius
	double	*d1_R_p, *d1_R_n;
	// gas pressure
	double	*PG_p, *PG_n;
	// liquid pressure
	double	*PL_p, *PL_n, *PL_m;
	double	*Q_B;
	// number weight of reference bubble
	double	*n_B;
	// time increment
	double	*dt, *dt_n;
	// remaining time
	double	*re, *re_n;
	// velocity of bubble
	double2	*v_B;
	// velocity of liquid
	double2	*v_L;
};

struct __align__(16) bubble_t_aos
{
	// index in mid cell
	int2	ibm;
	// index in node cell
	int2	ibn;
	// position
	double2	pos;
	// radius
	double	R_t;
	double	R_p, R_pn;
	double	R_n, R_nn;
	// time derivative of radius
	double	d1_R_p, d1_R_n;
	// gas pressure
	double	PG_p, PG_n;
	// liquid pressure
	double	PL_p, PL_n, PL_m;
	double	Q_B;
	// number weight of reference bubble
	double	n_B;
	// time increment
	double	dt, dt_n;
	// remaining time
	double	re, re_n;
	// velocity of bubble
	double2	v_B;
	// velocity of liquid
	double2	v_L;
};

struct __align__(16) solution_space
{
	double *p0;
	double *T;
	double2	*p, *pn;
	double	*vx, *vy;
	double	*c_sl;
	double	*rho_m, *rho_l;
	double	*f_g,*f_gn,*f_gm;
	double	*k_m, *C_pm;
	double	*Work;
	double	*Ex, *Ey;
	
	// index in mid cell
	int2	*ibm;
	// index in node cell
	int2	*ibn;
	// position
	double2	*pos;
	// radius
	double	*R_t;
	double	*R_p, *R_pn;
	double	*R_n, *R_nn;
	// time derivative of radius
	double	*d1_R_p, *d1_R_n;
	// gas pressure
	double	*PG_p, *PG_n;
	// liquid pressure
	double	*PL_p, *PL_n, *PL_m;
	double	*Q_B;
	// number weight of reference bubble
	double	*n_B;
	// time increment
	double	*dt, *dt_n;
	// remaining time
	double	*re, *re_n;
	// velocity of bubble
	double2	*v_B;
	// velocity of liquid
	double2	*v_L;
};

thrust::host_vector<solution_space> solve_bubbles(grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, bub_params_t *bub_params, plane_wave_t *plane_wave, debug_t *debug, int argc, char ** argv);

int initialize_variables (grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, plane_wave_t *plane_wave, array_index_t *array_index, mix_params_t *mix_params, bub_params_t *bub_params);

int initialize_CUDA_variables(grid_t *grid_size, PML_t *PML, sim_params_t *sim_params, plane_wave_t *plane_wave, array_index_t *array_index, mix_params_t *mix_params, bub_params_t *bub_params);
int destroy_CUDA_variables();

grid_t init_grid_size(grid_t grid_size);

plane_wave_t init_plane_wave(plane_wave_t plane_wave, grid_t grid_size);

array_index_t init_array (const grid_t grid_size, const sim_params_t sim_params);

bub_params_t	init_bub_params (bub_params_t bub_params, sim_params_t sim_params, double dt);

grid_gen init_grid_vector(array_index_t array_index, grid_t grid_size);

sigma_t init_sigma(const PML_t PML, const sim_params_t sim_params, const grid_t grid_size, const array_index_t array_index);
mix_params_t init_mix();

double mix_set_time_increment(sim_params_t sim_params, double dx_min, double u_max);
mixture_t init_mix_array(mix_params_t *mix_params, array_index_t array_index);

bubble_t init_bub_array(bub_params_t *bub_params, mix_params_t *mix_params, array_index_t *array_index, grid_t *grid_size, plane_wave_t *plane_wave);
bubble_t_aos bubble_input(double2 pos, double fg_in, bub_params_t bub_params, grid_t grid_size, plane_wave_t plane_wave);

void checkCUDAError(const char* msg);
#endif


