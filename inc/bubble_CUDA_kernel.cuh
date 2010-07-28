#include "bubble_CUDA.h"

// Array sizes for mixture and bubble variables, used in kernel dimension declarations
extern int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;
extern int numBubbles;

/* Constant Memory */
__device__ __constant__	double		Pi;
__device__ __constant__	double		Pi4r3;

__device__ __constant__	mixture_t	mixture_c;
__device__ __constant__	bubble_t	bubbles_c;

__device__ __constant__	sigma_t		sigma_c; 	// Sigma coefficients used in PML calculations
__device__ __constant__	grid_gen	gridgen_c; 	// Arrays containing rxp and xu, used in cylindrical coordinate calculations

// Parameters
__device__ __constant__	array_index_t	array_c;	// Array indices
__device__ __constant__	int		num_bubbles;	// Number of bubbles
__device__ __constant__ grid_t 		grid_c;		// Simulation domain size
__device__ __constant__ PML_t 		PML_c;		// PML
__device__ __constant__	sim_params_t	sim_params_c;	// Simulation parameters
__device__ __constant__	mix_params_t	mix_params_c;	// Mixture
__device__ __constant__	bub_params_t	bub_params_c;	// Bubble
__device__ __constant__	plane_wave_t	plane_wave_c;	// Plane Wave
__device__ __constant__	double		tstep_c;	// Current thetime


/* Forward Declarations */

// Material properties of water
__inline__ __host__ __device__ double density_water(double, double);	// Density
__inline__ __host__ __device__ double thermal_expansion_rate_water(double, double);	// Rate of thermal Expansion
__inline__ __host__ __device__ double adiabatic_sound_speed_water(double, double);	// Speed of propagation of sound
__inline__ __host__ __device__ double viscosity_water(double);	// Viscosity
__inline__ __host__ __device__ double thermal_conductivity_water(double);	// Thermal conductivity
__inline__ __host__ __device__ double specific_heat_water(double);	// Specific heat capacity
__inline__ __host__ __device__ double vapor_pressure_water(double);	// Vapor pressure
__inline__ __host__ __device__ double surface_tension_water(double);	// Surface tension

// Smooth delta function
__inline__ __host__ __device__ double smooth_delta_x(const int, const double);	// x-direction (includes cylindrical case)
__inline__ __host__ __device__ double smooth_delta_y(const int, const double);	// y-direction
__inline__ __host__ __device__ double fg_from_bubble(const int2, const double2, const double, const double);	// Calculates weighted void fraction based on the smooth delta function

// Functions used by the bubble radius solver
__inline__ __host__ __device__ void solveRayleighPlesset(double * , double * , double * , double * , const double * , const double * , double * , double *, bub_params_t);	// Solves the Rayleigh Plesset equation
__inline__ __host__ __device__ double solveOmegaN(doublecomplex *, double, double, bub_params_t);	// Solves for omega_N
static __inline__ __host__ __device__ doublecomplex Alpha(double, double, double);	// Calculates alpha_N
static __inline__ __host__ __device__ doublecomplex Upsilon(doublecomplex , double);	// Calculates Upsilon
static __inline__ __host__ __device__ doublecomplex solveLp(doublecomplex, double);	// Calculates Lp_N
__inline__ __host__ __device__ double solvePG(double, double, double, double, double, doublecomplex, bub_params_t);	// Calculates PG


/* Static utility functions */

// Implements atomic addition for doubles
static __inline__ __device__ double atomicAdd(double * addr, double val){
    double old = *addr;
    double assumed;

    do {
        assumed = old;
        old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
                                        __double_as_longlong(assumed),
                                        __double_as_longlong(val+assumed)));	// Pull down down the variable, and compare and swap when we have the chance. Note that atomicCAS returns the new value at addr.
    } while( assumed!=old );

    return old;	// We return the final value, to conform with other atomicAdd() implementations
}

// intrinsic epsilon for double
static __inline__ __host__ __device__ double epsilon(double val){
	return 2.22044604925031308e-016;
	//return 2.0e-16;
	//return 1.0e-10;
}

// Implements hyperbolic cotangent for double precision complex variables
static __inline__ __host__ __device__ doublecomplex complexcoth(doublecomplex z){
	double x = tanh(z.real()), y = tan(z.imag());
	return make_doublecomplex(1.0, x * y)/make_doublecomplex(x,y);
}

/* Reduced Order Bubble Dynamics Modelling Functions */

// Rayleigh Plesset solver
__inline__ __host__ __device__ void solveRayleighPlesset(double * Rt, double *Rp, double * Rn, double * d1R, const double * PG, const double * PL, double * dt, double * remain, bub_params_t bub_params){
	double 	Rm, d2R, 	// Temporary radius variables
		dtm, dtp, 	// Temporary time variables
		c1, c2, c3;	// Registers

	Rm = *Rn;
	*Rn = *Rp;

	d2R = (( *PG - *PL - 2.0*bub_params.sig/(*Rn) - 4.0*bub_params.mu/(*Rn)*(*d1R) )/bub_params.rho - 1.5*(*d1R)*(*d1R)) / (*Rn);

	dtm = *dt;
	dtp = min(1.01 * (*dt), bub_params.dt0);

	c1 = abs(*d1R);
	c2 = abs(d2R);

	if (c1 > epsilon(c1)){
		dtp = min(dtp, 0.01 * (*Rn)/c1);
	}
	if (c2 > epsilon(c2)){
		dtp = min(dtp, sqrt(0.01 * (*Rn)/c2));
	}

	c2 = -dtp/dtm;
	c1 = 1.0 - c2;
	c3 = dtp * 0.5 * (dtp + dtm);
	*Rp = c1 * (*Rn) + c2 * Rm + c3 * d2R;

	c1 = dtm*dtm;
	c3 = -dtp*dtp;
	c2 = -c1-c3;
	*d1R = (c1 * (*Rp) + c2 * (*Rn) + c3 * Rm)/(dtm * dtp * (dtm + dtp)) + dtp * d2R;

	*dt = dtp;
	*Rt = *Rp;

	if (dtp >= *remain){
		dtp = *remain;
		c2 = -dtp/dtm;
		c1 = 1.0 - c2;
		c3 = dtp * 0.5 * (dtp+dtm);
		*Rt = c1 * (*Rn) + c2 * Rm + c3 * d2R;
	}
	*remain -= dtp;
	return;
}

// Implicit solver for omega_N and alpha_N
__inline__ __host__ __device__ double solveOmegaN(doublecomplex * alpha_N, double PG, double R, bub_params_t bub_params){
	double R2 = R * R;
	double coef1 = (bub_params.PG0 * bub_params.R03)/(PG * R * R2);
	double coef2 = PG / (bub_params.rho * R2);
	double coef3 = 4.0 / (bub_params.rho * bub_params.rho * R2 * R2);
	double value1 = -2.0 * bub_params.sig/(bub_params.rho * R2 * R);

	double omega_N;

	// Zeroeth step
	doublecomplex Upsilon_N = make_doublecomplex(3.0*bub_params.gam, 0.0) * coef1;
	double mu_eff = bub_params.mu;
	double eta = Upsilon_N.real() * coef2 + value1 - mu_eff * mu_eff * coef3;

	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	Upsilon_N = Upsilon(*alpha_N, bub_params.gam) * coef1;
	mu_eff = bub_params.mu + Upsilon_N.imag() * PG / (4.0 * (omega_N));
	eta = Upsilon_N.real() * coef2 + value1 - mu_eff * mu_eff * coef3;

	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	Upsilon_N = Upsilon(*alpha_N, bub_params.gam) * coef1;
	mu_eff = bub_params.mu + Upsilon_N.imag() * PG / (4.0 * (omega_N));
	eta = Upsilon_N.real() * coef2 + value1 - mu_eff * mu_eff * coef3;

	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	Upsilon_N = Upsilon(*alpha_N, bub_params.gam) * coef1;
	mu_eff = bub_params.mu + Upsilon_N.imag() * PG / (4.0 * (omega_N));
	eta = Upsilon_N.real() * coef2 + value1 - mu_eff * mu_eff * coef3;

	omega_N = sqrt(max(eta, 1.0e-6*epsilon(eta)));
	*alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
	
	return omega_N;
}

// returns alpha_N
static __inline__ __host__ __device__ doublecomplex Alpha(double R, double omega_N, double coeff_alpha){
	return make_doublecomplex(1.0, 1.0) * sqrt(coeff_alpha * (omega_N) / (R));
}

// returns Upsilon_N
static __inline__ __host__ __device__ doublecomplex Upsilon(doublecomplex a, double gam){
	doublecomplex ctmp;
	doublecomplex a2 = a * a;
	if (a.abs() > 1.0e-2){
		ctmp = (a * complexcoth(a) - 1.0)/(a2);
	}
	else{
		ctmp = 1.0/3.0 + a2 * (-1.0/45.0 + a2 * (2.0/945.0 - a2 / 4725.0));
	}
	return (3.0 * gam)/(1.0+3.0*(gam-1.0)*ctmp);
}

// returns Lp_N
static __inline__ __host__ __device__ doublecomplex solveLp(doublecomplex a, double R){
	doublecomplex ctmp;
	doublecomplex a2 = a * a;
	if(a.abs() > 1.0e-1){
		ctmp = a * complexcoth(a) - 1.0;
		ctmp = (a2 - 3.0 * ctmp)/(a2*ctmp);
	}
	else{
		ctmp = 1.0/5.0 + a2*(-1.0/175.0+a2*(2.0/7875.0 - a2*(37.0/3031875.0)));
	}

	return ctmp * R;
}

// returns PG
__inline__ __host__ __device__ double solvePG(double PGn, double Rp, double Rn, double omega_N, double dt, doublecomplex Lp_N, bub_params_t bub_params){
	double 	Lp_NR = Lp_N.real() / (Lp_N.abs() * Lp_N.abs()),
		Lp_NI = Lp_N.imag() / (Lp_N.abs() * Lp_N.abs()),
		R = 0.5 * (Rp + Rn);

	double	c0 = dt*3.0*(bub_params.gam-1.0) * 2.0/(Rp + Rn) * bub_params.K0 * bub_params.T0/(bub_params.PG0 * bub_params.R03),
		c3 = 3.0 * bub_params.gam/(Rp+Rn)*(Rp-Rn) + c0 * Lp_NR * 0.5 * R*R*R,
		c1 = 1.0 + c3 - c0*Lp_NI/(omega_N * dt) * Rp*Rp*Rp,
		c2 = 1.0 - c3 - c0*Lp_NI/(omega_N * dt) * Rn*Rn*Rn;

	return (c2*PGn + c0*Lp_NR * bub_params.PG0 * bub_params.R03)/c1;
}

/* Bubble Kernels */

// Updates the positional index of an array of bubbles
__global__ void BubbleUpdateIndexKernel(){
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const bool ok = index < num_bubbles;

	double2 pos;
	int2 ibm, ibn;

	// Cache positions to shared memory
	if (ok){
		pos = bubbles_c.pos[index];

		// Convert positions into indices
		ibm = make_int2(floor(pos.x * grid_c.rdx + 1.0), floor(pos.y * grid_c.rdy + 1.0));
		ibn = make_int2(floor(pos.x * grid_c.rdx + 0.5), floor(pos.y * grid_c.rdy + 0.5));

		// Store indices
		bubbles_c.ibm[index] = ibm;
		bubbles_c.ibn[index] = ibn;
	}
}

// Performs interpolation on surrounding mixture cells to determine the pressure acting on a bubble
__global__ void BubbleInterpolationScalarKernel(int p0_width){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	int2 ibn;
	double2 pos;
	double Delta_x, Delta_y;
	double PL_p;

	if(index < num_bubbles){
		// Cache to shared memory
		ibn = bubbles_c.ibn[index];
		pos = bubbles_c.pos[index];
		PL_p = 0.0;

		// Run a search around the vicinity of the bubble
		for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++){
			Delta_x = smooth_delta_x(i,pos.x);	// Compute the x component of smooth delta function
			for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++){
				Delta_y = smooth_delta_y(j, pos.y);	// Compute the y component of smooth delta function
				if(	(i >= array_c.ista1m) &&
					(i <= array_c.iend1m) &&
					(j >= array_c.jsta1m) &&
					(j <= array_c.jend1m)){		// Check against grid boundaries
					PL_p += mixture_c.p0[p0_width * (j - array_c.jsta1m) + (i - array_c.ista1m)] * Delta_x * Delta_y;	// Increment liquid pressure according to influence function (smooth delta)
				}
			}
		}
		bubbles_c.PL_p[index] = PL_p; // Store total liquid pressure
	}
}

// Performs interpolation on surrounding mixture cells to determine the velocity of a bubble
__global__ void BubbleInterpolationVelocityKernel(int vx_width, int vy_width){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	int2 ibm, ibn;
	double2 pos;
	double2 v_L = make_double2(0.0, 0.0);
	double Delta_x, Delta_y;

	if (index < num_bubbles){
		// Cache into shared memory
		ibm = bubbles_c.ibm[index];
		ibn = bubbles_c.ibn[index];
		pos = bubbles_c.pos[index];

		// x-component of velocity
		for (int i = ibm.x + bub_params_c.nbs; i <= ibm.x + bub_params_c.nbe; i++){
			Delta_x = smooth_delta_x(i,pos.x);	// compute x-component of smooth delta function
			for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++){
				Delta_y = smooth_delta_y(j,pos.y);	// compute y-component of smooth delta function
				if(	(i >= array_c.ista2n) &&
					(i <= array_c.iend2n) &&
					(j >= array_c.jsta2m) &&
					(j <= array_c.jend2m)){		// check against boundaries
					v_L.x += mixture_c.vx[vx_width * (j - array_c.jsta2m) + (i - array_c.ista2n)] * Delta_x * Delta_y; // increment velocity
				}
			}
		}
		// y-component of velocity
		for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++){
			Delta_x = smooth_delta_x(i,pos.x);	// compute x-component of smooth delta function
			for (int j = ibm.y + bub_params_c.nbs; j <= ibm.y + bub_params_c.nbe; j++){
				Delta_y = smooth_delta_y(j,pos.y);	// compute y-component of smooth delta function
				if(	(i >= array_c.ista2m) &&
					(i <= array_c.iend2m) &&
					(j >= array_c.jsta2n) &&
					(j <= array_c.jend2n)){		// check against boundaries
					v_L.y += mixture_c.vy[vy_width * (j - array_c.jsta2n) + (i - array_c.ista2m)] * Delta_x * Delta_y; // increment velocity
				}
			}
		}
		bubbles_c.v_L[index] = v_L; // store the total velocity
	}
}


// Determines the translational velocity of a bubble
// This is a placeholder, in case we abandon the no-slip condition
__global__ void BubbleTranslationMotionKernel(){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	const bool ok = index < num_bubbles;

	double2 v_B;

	if (ok){
		v_B = bubbles_c.v_L[index];
	}
	__syncthreads();
	if (ok){
		bubbles_c.v_B[index] = v_B;
	}
}

// 	Calculates the positional movement of a bubble based on it's velocity
__global__ void BubbleMotionKernel(){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	const bool ok = index < num_bubbles;

	double2 v_B;
	double2 pos;
	//double2 pos0 = make_double2(0.0, 0.0);					// X0 and Y0 boundary
	//double2 pos1 = make_double2(grid_c.X * grid_c.dx, grid_c.Y * grid_c.dy); 	// X1 and Y1 boundary

	if (ok){
		pos = bubbles_c.pos[index];
		v_B = bubbles_c.v_B[index];

		pos.x += v_B.x * mix_params_c.dt;
		pos.y += v_B.y * mix_params_c.dt;
		bubbles_c.pos[index] = pos;
	}
}

// Solves the Rayleigh-Plesset equations for bubble dynamics, to calculate bubble radius
__global__ void BubbleRadiusKernel(int * max_iter){
	// double d1Rn;
	double PGn, PL, Rt, omega_N;
	double dTdr_R, SumHeat = 0.0, SumVis = 0.0;
	doublecomplex alpha_N, Lp_N;

	double Rp, Rn, d1Rp, PGp, dt_L, remain, PC0, PC1, PC2, thetime;

	double temp[3];

	for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < num_bubbles + gridDim.x * blockDim.x; index += gridDim.x * blockDim.x){
	if (index < num_bubbles){
		Rp 	= bubbles_c.R_pn[index];
		Rn 	= bubbles_c.R_nn[index];
		d1Rp 	= bubbles_c.d1_R_n[index];
		PGp	= bubbles_c.PG_n[index];
		dt_L	= bubbles_c.dt_n[index];
		remain	= bubbles_c.re_n[index] + mix_params_c.dt;

		PC0 = bubbles_c.PL_n[index] + bub_params_c.PL0;
		PC1 = 0.5*(bubbles_c.PL_p[index]-bubbles_c.PL_m[index])/mix_params_c.dt;
		PC2 = 0.5*(bubbles_c.PL_p[index]+bubbles_c.PL_m[index]-2.0*bubbles_c.PL_n[index])/(mix_params_c.dt * mix_params_c.dt);

		thetime = -bubbles_c.re_n[index];

		max_iter[index] = 0;
	}

	if (index < num_bubbles){
		while (remain > 0.0){
			//d1Rn 	= 	d1Rp;
			PGn 	= 	PGp;
			PL 	= 	PC2 * thetime * thetime + PC1 * thetime + PC0;


			solveRayleighPlesset(&Rt, &Rp, &Rn, &d1Rp, &PGn, &PL, &dt_L, &remain, bub_params_c);
			thetime = thetime + dt_L;

			omega_N = solveOmegaN (&alpha_N, PGn, Rn, bub_params_c);


			Lp_N = solveLp(alpha_N, Rn);

			PGp = solvePG(PGn, Rp, Rn, omega_N, dt_L, Lp_N, bub_params_c);

			temp[0] = 0.5*(Rp+Rn);
			temp[1] = 1 / (Lp_N.abs() * Lp_N.abs()) * bub_params_c.T0 / (bub_params_c.PG0 * bub_params_c.R03);
			dTdr_R 	= 	Lp_N.real() * temp[1] * (bub_params_c.PG0 * bub_params_c.R03 - 0.5*(PGp+PGn)*temp[0]*temp[0]*temp[0]) +
					Lp_N.imag() * temp[1] * (PGp * Rp * Rp * Rp - PGn * Rn * Rn * Rn) / (omega_N * dt_L);

			SumHeat -= 4.0 * Pi * Rp * Rp * bub_params_c.K0 * dTdr_R * dt_L;
			SumVis 	+= 4.0 * Pi * Rp * Rp * 4.0 * bub_params_c.mu * d1Rp / Rp * d1Rp * dt_L;

			max_iter[index]++;
		}
	}

	if (index < num_bubbles){
		// Assign values back to the global memory
		bubbles_c.R_t[index] 	= Rt;
		bubbles_c.R_p[index] 	= Rp;
		bubbles_c.R_n[index] 	= Rn;
		bubbles_c.d1_R_p[index]	= d1Rp;
		bubbles_c.PG_p[index] 	= PGp;
		bubbles_c.dt[index]	= dt_L;
		bubbles_c.re[index]	= remain;
		bubbles_c.Q_B[index]	= (SumHeat + SumVis) / (mix_params_c.dt - remain + bubbles_c.re_n[index]);

		//}
	}
	}
}

/* Mixture Kernels */

// Perform relevant void fraction operations on a cylindrical grid
__global__ void VoidFractionCylinderKernel(int fg_width){
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int j2m = j + array_c.jsta2m;

	if ((i == 0) && (j2m >= array_c.jsta2m) && (j2m <= array_c.jend2m)){
		for (int count = 1 + array_c.ns + array_c.ms; count <= 0; count ++){
			// Mirror void fractions along the center axis
			mixture_c.f_g[fg_width * j + (count - array_c.ista2m)] += mixture_c.f_g[fg_width * j + (1 - count - array_c.ista2m)];
			mixture_c.f_g[fg_width * j + (1 - count - array_c.ista2m)] = mixture_c.f_g[fg_width * j + (count - array_c.ista2m)];
		}
	}
}

// Calculates the void fraction from bubbles
__global__ void VoidFractionReverseLookupKernel(int fg_width){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	int2 ibn;
	double2 pos;
	double n_B;
	double R_t;
	double fg;

	if (index < num_bubbles){
		ibn = bubbles_c.ibn[index];
		pos = bubbles_c.pos[index];
		n_B = bubbles_c.n_B[index];
		R_t = bubbles_c.R_t[index];
		for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++){
			for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++){
				if((i >= array_c.ista2m) && (i <= array_c.iend2m) && (j >= array_c.jsta2m) && (j <= array_c.jend2m)){
					fg = fg_from_bubble(make_int2(i,j), pos, n_B, R_t);
					atomicAdd(&mixture_c.f_g[(j - array_c.jsta2m) * fg_width + (i - array_c.ista2m)], fg);
				}
			}
		}
	}
}

// Predicts the void fraction at step n + 1, and stores it in the work array
__global__ void VFPredictionKernel(int fg_width){
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int i2m = i + array_c.ista2m;
	const int j2m = j + array_c.jsta2m;
	const int fg_i = fg_width * j + i;

	// Load void fractions for n, n+1, n-1
	if ((i2m <= array_c.iend2m) && (j2m <= array_c.jend2m)){
		mixture_c.Work[fg_i] = 3.0 * mixture_c.f_g[fg_i] - 3.0 * mixture_c.f_gn[fg_i] + mixture_c.f_gm[fg_i]; // Calculate future void fraction and store into work array
	}
}

__global__ void VelocityKernel(int vx_width, int vy_width, int rhom_width, int p0_width, int csl_width){
	__shared__ double p0[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH + 1];
	__shared__ double rhom[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH + 1];
	__shared__ double csl[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH + 1];
	
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = blockDim.x * blockIdx.x + tx;
	const int j = blockDim.y * blockIdx.y + ty;
	const int in = i + array_c.ista2n;
	const int jn = j + array_c.jsta2n;
	
	const int rhom_i= rhom_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
	const int p0_i	= p0_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
	const int csl_i = csl_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
	const int vx_i =  vx_width * (jn - array_c.jsta2m) + (in - array_c.ista2n);
	const int vy_i =  vy_width * (jn - array_c.jsta2n) + (in - array_c.ista2m);
	
	bool ok1 = (in >= array_c.istan) && (in <= array_c.iendn) && (jn >= array_c.jstam) && (jn <= array_c.jendm);
	bool ok2 = (in >= array_c.istam) && (in <= array_c.iendm) && (jn >= array_c.jstan) && (jn <= array_c.jendn);
	
	double s0 = 0.5 * mix_params_c.dt;
	double s1, s2, s3;

	if (ok1 || ok2){
		p0[ty][tx] = mixture_c.p0[p0_i];
		rhom[ty][tx] = mixture_c.rho_m[rhom_i];
		csl[ty][tx] = mixture_c.c_sl[csl_i];
	}
	if (ok1 && ((tx == blockDim.x - 1) || (in == array_c.iendn))){
		p0[ty][tx + 1] = mixture_c.p0[p0_i + 1];
		rhom[ty][tx + 1] = mixture_c.rho_m[rhom_i + 1];
		csl[ty][tx + 1] = mixture_c.c_sl[csl_i + 1];
	}
	if (ok2 && ((ty == blockDim.y - 1)||(jn == array_c.jendn))){
		rhom[ty + 1][tx] = mixture_c.rho_m[rhom_i + rhom_width];
		p0[ty + 1][tx] = mixture_c.p0[p0_i + p0_width];
		csl[ty + 1][tx] = mixture_c.c_sl[csl_i + csl_width];
	}
	__syncthreads();
	if (ok1){
		s1 = (rhom[ty][tx] + rhom[ty][tx + 1]) * 0.5;
		s2 = (-p0[ty][tx] + p0[ty][tx + 1]) * grid_c.rdx;
		s3 = (csl[ty][tx] + csl[ty][tx + 1]) * 0.5;
		s3 = s3 * s0 * sigma_c.nx[i];
		mixture_c.vx[vx_i] = (mixture_c.vx[vx_i] * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);
	}
	if (ok2){
		s1 = ( rhom[ty][tx] + rhom[ty + 1][tx] ) * 0.5;
		s2 = (-p0[ty][tx] + p0[ty + 1][tx] ) * grid_c.rdx;
		s3 = ( csl[ty][tx] + csl[ty + 1][tx] ) * 0.5;
		s3 = s3 * s0 * sigma_c.ny[j];
		mixture_c.vy[vy_i] = (mixture_c.vy[vy_i] * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);
	}
}

//	Calculates the x-component of velocity of the mixture field
__global__ void VXKernel(int vx_width, int rhom_width, int p0_width, int csl_width){
	__shared__ double p0[TILE_BLOCK_HEIGHT][TILE_BLOCK_WIDTH + 1];
	__shared__ double rhom[TILE_BLOCK_HEIGHT][TILE_BLOCK_WIDTH + 1];
	__shared__ double csl[TILE_BLOCK_HEIGHT][TILE_BLOCK_WIDTH + 1];

	double vx;

	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = blockDim.x * blockIdx.x + tx;
	const int j = blockDim.y * blockIdx.y + ty;
	const int in = i + array_c.ista2n;
	const int jm = j + array_c.jsta2m;

	const int rhom_i= rhom_width * (jm - array_c.jsta1m) + (in - array_c.ista1m);
	const int p0_i	= p0_width * (jm - array_c.jsta1m) + (in - array_c.ista1m);
	const int csl_i = csl_width * (jm - array_c.jsta1m) + (in - array_c.ista1m);
	const int vx_i =  vx_width * (jm - array_c.jsta2m) + (in - array_c.ista2n);

	bool oksave = (in >= array_c.istan) && (in <= array_c.iendn) && (jm >= array_c.jstam) && (jm <= array_c.jendm);
	double s0 = 0.5 * mix_params_c.dt;
	double s1, s2, s3;

	if (oksave){
		vx = mixture_c.vx[vx_i];

		p0[ty][tx] = mixture_c.p0[p0_i];
		rhom[ty][tx] = mixture_c.rho_m[rhom_i];
		csl[ty][tx] = mixture_c.c_sl[csl_i];
	}
	if (oksave && ((tx == blockDim.x - 1) || (in == array_c.iendn))){
		p0[ty][tx + 1] = mixture_c.p0[p0_i + 1];
		rhom[ty][tx + 1] = mixture_c.rho_m[rhom_i + 1];
		csl[ty][tx + 1] = mixture_c.c_sl[csl_i + 1];
	}
	__syncthreads();
	if (oksave){
		s1 = (rhom[ty][tx] + rhom[ty][tx + 1]) * 0.5;
		s2 = (-p0[ty][tx] + p0[ty][tx + 1]) * grid_c.rdx;
		s3 = (csl[ty][tx] + csl[ty][tx + 1]) * 0.5;
		s3 = s3 * s0 * sigma_c.nx[i];
		vx = (vx * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);
	}
	__syncthreads();
	if (oksave){
		mixture_c.vx[vx_width * (jm-array_c.jsta2m) + (in - array_c.ista2n)] = vx;
	}

}

//	Calculates the velocity at the boundary
//	X-direction only
__global__ void VXBoundaryKernel(int vx_width){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	const int i2n = index + array_c.ista2n;
	const int j2m = index + array_c.jsta2m;

	bool ista2n_to_iend2n = (i2n >= array_c.ista2n) && (i2n <= array_c.iend2n);
	bool jstam_to_jendm = (j2m >= array_c.jstam) && (j2m <= array_c.jendm);

	double vx;

	if (PML_c.X0 == 0){
		if (jstam_to_jendm){
			for (int i = array_c.ms + array_c.ns; i <= -1; i++){
				mixture_c.vx[vx_width * index + i - array_c.ista2n] = mixture_c.vx[vx_width * index - i - array_c.ista2n];
			}
			mixture_c.vx[vx_width * (index) + 0 - array_c.ista2n] = 0.0;
		}
	}
	__syncthreads();
	if (PML_c.X1 == 0){
		if (jstam_to_jendm){
			for (int i = grid_c.X + 1; i <= grid_c.X + array_c.me + array_c.ne; i++){
				vx = -mixture_c.vx[vx_width * (index) + grid_c.X - (i - grid_c.X) - array_c.ista2n];
				mixture_c.vx[vx_width * (index) + i - array_c.ista2n] = vx;
			}
			mixture_c.vx[vx_width * (index) + grid_c.X - array_c.ista2n] = 0.0;
		}
	}
	__syncthreads();
	if (ista2n_to_iend2n){
		for (int j = 1 + array_c.ms + array_c.ns; j <= 0; j++){
			mixture_c.vx[vx_width * (j - array_c.jsta2m) + index] = mixture_c.vx[vx_width * ( 1 - j - array_c.jsta2m) + index];
		}
		for (int j = grid_c.Y + 1; j <= grid_c.Y + array_c.me + array_c.ne; j++){
			mixture_c.vx[vx_width * (j - array_c.jsta2m) + index] = mixture_c.vx[vx_width * (grid_c.Y - (j - grid_c.Y - 1) - array_c.jsta2m) + index];;
		}

	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the velocity of the mixture field
//	Y-direction only
//

__global__ void VYKernel(int vy_width, int rhom_width, int p0_width, int csl_width){
	__shared__ double rho_m[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH];
	__shared__ double p0[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH];
	__shared__ double c_sl[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH];

	double vy;

	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx*blockDim.x + tx;
	const int j = by*blockDim.y + ty;
	const int im = i + array_c.ista2m;
	const int jn = j + array_c.jsta2n;


	const int rhom_i= rhom_width * (jn - array_c.jsta1m) + (im - array_c.ista1m);
	const int p0_i	= p0_width * (jn - array_c.jsta1m) + (im - array_c.ista1m);
	const int csl_i = csl_width * (jn - array_c.jsta1m) + (im - array_c.ista1m);
	const int vy_i =  vy_width * (jn - array_c.jsta2n) + (im - array_c.ista2m);

	bool oksave = (im >= array_c.istam) && (im <= array_c.iendm) && (jn >= array_c.jstan) && (jn <= array_c.jendn);
	double s0 = 0.5 * mix_params_c.dt;
	double s1, s2, s3;



	if (oksave){
		rho_m[ty][tx]	= mixture_c.rho_m[rhom_i];
		p0[ty][tx]	= mixture_c.p0[p0_i];
		c_sl[ty][tx]	= mixture_c.c_sl[csl_i];
	}
	if (oksave && ((ty == blockDim.y - 1)||(jn == array_c.jendn))){
		rho_m[ty + 1][tx] = mixture_c.rho_m[rhom_i + rhom_width];
		p0[ty + 1][tx] = mixture_c.p0[p0_i + p0_width];
		c_sl[ty + 1][tx] = mixture_c.c_sl[csl_i + csl_width];
	}
	if (oksave){
		vy = mixture_c.vy[vy_i];
	}
	__syncthreads();
	if (oksave){
//		s1 = (mixture_c.rho_m[rhom_i] + mixture_c.rho_m[rhom_i + rhom_width]) * 0.5;
//		s2 = (-mixture_c.p0[p0_i] + mixture_c.p0[p0_i + p0_width]) * grid_c.rdx;
//		s3 = (mixture_c.c_sl[csl_i] + mixture_c.c_sl[csl_i + csl_width]) * 0.5;
		s1 = ( rho_m[ty][tx] + rho_m[ty + 1][tx] ) * 0.5;
		s2 = (-p0[ty][tx] + p0[ty + 1][tx] ) * grid_c.rdx;
		s3 = ( c_sl[ty][tx] + c_sl[ty + 1][tx] ) * 0.5;
		s3 = s3 * s0 * sigma_c.ny[j];
		vy = (vy * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);

	}
	__syncthreads();
	if (oksave){
		mixture_c.vy[vy_i] = vy;
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the velocity at the boundary
//	Y-direction only
//

__global__ void VYBoundaryKernel(int vy_width){
	const int BLOCK_SIZE = LINEAR_BLOCK_SIZE;
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;

	const int i = bx*BLOCK_SIZE + tx;

	const int i2m = i + array_c.ista2m;
	const int j2n = i + array_c.jsta2n;

	double s1, s2, xm, ym;

	bool istam_to_iendm = ((i2m >= array_c.istam) && (i2m <= array_c.iendm));
	bool jsta2n_to_jend2n = ((j2n >= array_c.jsta2n) && (j2n <= array_c.jend2n));

	double vy;

	if (PML_c.Y0 == 0){
		if (istam_to_iendm){
			if ((plane_wave_c.Plane_P == 0) && (plane_wave_c.Focused_P == 0)){
				mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i] = 0.0;
			}

			for (int count = array_c.ns + array_c.ms; count <= -1; count++){
				vy = mixture_c.vy[vy_width*(-(count+1) - array_c.jsta2n) + i];
				mixture_c.vy[vy_width*(count - array_c.jsta2n) + i] = vy;
			}

			if (plane_wave_c.Plane_V == 1){
				if ((fmod(tstep_c, (((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq))
					<=
					((double) plane_wave_c.on_wave)/plane_wave_c.freq)){
					mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i]
						= plane_wave_c.amp * sin(plane_wave_c.omega * tstep_c);
				}
				else{
					mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i] = 0.0;
				}
			}
			if(plane_wave_c.Focused_V == 1){
				ym = 0.0;
				xm = ((double)i2m - 0.5) * grid_c.dx;
				s1 = sqrt(	((plane_wave_c.fp.x - xm)*(plane_wave_c.fp.x - xm)) +
						((plane_wave_c.fp.y - ym)*(plane_wave_c.fp.y - ym)));
				if (s1 <= plane_wave_c.f_dist){
					s1 = max(tstep_c - (plane_wave_c.f_dist - s1) / mix_params_c.cs_inf, 0.0);
					s2 = (fmod(s1,((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq));
					if (s2 < ((double)plane_wave_c.on_wave/plane_wave_c.freq)){
						mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i] = plane_wave_c.amp * sin(plane_wave_c.omega * s2);
					}
					else{
						mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i] = 0.0;
					}
				}
				else{
					mixture_c.vy[vy_width*(0 - array_c.jsta2n) + i] = 0.0;
				}

			}
		}
	}
	if (PML_c.Y1 == 0){
	if (istam_to_iendm){
		mixture_c.vy[vy_width*(grid_c.Y-array_c.jsta2n) + i] = 0.0;
		for (int count = grid_c.Y + 1; count <= grid_c.Y + array_c.ne + array_c.me; count++){
			vy = mixture_c.vy[vy_width*(grid_c.Y - (count - grid_c.Y) - array_c.jsta2n) + i];
			mixture_c.vy[vy_width*(count - array_c.jsta2n) + i] = vy;
		}
	}
	}
	if (jsta2n_to_jend2n){
		for (int count = 1 + array_c.ms + array_c.ns; count <= 0; count++){
			vy = mixture_c.vy[vy_width*i + 1 - count - array_c.ista2m];
			mixture_c.vy[vy_width*i + count - array_c.ista2m] = vy;
		}
		for (int count = grid_c.X + 1; count <= grid_c.X + array_c.me + array_c.ne; count++){
			vy = mixture_c.vy[vy_width*i + grid_c.X - (count - grid_c.X - 1) - array_c.ista2m];
			mixture_c.vy[vy_width*i + count - array_c.ista2m] = vy;
		}
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the mixture pressure
//
__global__ void MixturePressureKernel(int vx_width, int vy_width, int fg_width, int rhol_width, int csl_width, int p0_width, int p_width, int Work_width){
	__shared__ double xu[TILE_BLOCK_WIDTH + 1];
	__shared__ double vx[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH + 1];
	__shared__ double vy[TILE_BLOCK_HEIGHT + 1][TILE_BLOCK_WIDTH + 1];
	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx*blockDim.x + tx;
	const int j = by*blockDim.y + ty;
	const int i1m = i + array_c.ista1m;
	const int j1m = j + array_c.jsta1m;

	bool ok = ((i1m >= array_c.istam) && (i1m <= array_c.iendm) && (j1m >= array_c.jstam) && (j1m <= array_c.jendm));

	double s1, s2, s3, s4, s5, s6, s7;

	double f_g, f_gn;
	double rho_l, c_sl;

	double2 p, pn;

	// Keep in mind now that when we say xu[tx + 1], we really mean xu[tx], and so on

	if (ok){
		f_g = mixture_c.f_g[fg_width * (j1m - array_c.jsta2m) + (i1m - array_c.ista2m)];
		f_gn = mixture_c.f_gn[fg_width * (j1m - array_c.jsta2m) + (i1m - array_c.ista2m)];

		rho_l = mixture_c.rho_l[rhol_width * j + i];
		c_sl = mixture_c.c_sl[csl_width * j + i];

		pn = mixture_c.pn[p_width * j + i];

		vx[ty][tx + 1] = mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - array_c.ista2n)];
		vy[ty + 1][tx] = mixture_c.vy[vy_width * (j1m - array_c.jsta2n) + (i1m - array_c.ista2m)];

		if ((tx == 0) || (i1m == array_c.istam)){
			vx[ty][tx] = mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - 1 - array_c.ista2n)];
		}
		if ((ty == 0) || (j1m == array_c.jstam)){
			vy[ty][tx] = mixture_c.vy[vy_width * (j1m - 1 - array_c.jsta2n) + (i1m - array_c.ista2m)];
		}
		if (ty == 0 || (j1m == array_c.jstam)){
			xu[tx + 1] = gridgen_c.xu[i1m - array_c.ista2n];
			if ((tx == 0) || (i1m == array_c.istam)){
				xu[tx] = gridgen_c.xu[i1m - 1 - array_c.ista2n];
			}
		}
	}
	__syncthreads();

	if (ok){
	#ifdef _CYLINDRICAL_
		s1 = (-xu[tx]*vx[ty][tx] + xu[tx+1]*vx[ty][tx+1]) * grid_c.rdx * gridgen_c.rxp[i1m - array_c.ista2m];
//		s1 = 	(
//				-gridgen_c.xu[i1m - 1 - array_c.ista2n] * mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - 1 - array_c.ista2n)]
//				+ gridgen_c.xu[i1m - array_c.ista2n] * mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - array_c.ista2n)]
//			)
//			* grid_c.rdx * gridgen_c.rxp[i1m - array_c.ista2m];
	#else
		s1 = (-vx[ty][tx] + vx[ty][tx+1])*grid_c.rdx;
//		s1 = (-mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - 1 - array_c.ista2n)]
//			+ mixture_c.vx[vx_width * (j1m - array_c.jsta2m) + (i1m - array_c.ista2n)]) * grid_c.rdx;
	#endif
//		s2 = (-mixture_c.vy[vy_width * (j1m - 1 - array_c.jsta2n) + (i1m - array_c.ista2m)] + mixture_c.vy[vy_width * (j1m - array_c.jsta2n) + (i1m - array_c.ista2m)]) * grid_c.rdy;
		s2 = (-vy[ty][tx] + vy[ty+1][tx])*grid_c.rdy;
		s7 = -(f_g - f_gn)/ mix_params_c.dt / (1.0-0.5*(f_g + f_gn));
		s3 = mix_params_c.dt * rho_l * c_sl * c_sl;
		s4 = 0.5 * mix_params_c.dt * c_sl;
		s5 = s4 * sigma_c.mx[i];
		s6 = s4 * sigma_c.my[j];
		p = make_double2((pn.x * (1.0 - s5) - s3 * (s1 + 0.5 * s7)) / (1.0 + s5), (pn.y * (1.0 - s6) - s3 * (s2 + 0.5 * s7)) / (1.0 + s6));

		mixture_c.p[p_width * j + i] = p;

		mixture_c.Work[Work_width * j + i] = abs(mixture_c.p0[p0_width * j + i] - p.x - p.y);
		mixture_c.p0[p0_width * j + i] = p.x + p.y;
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the mixture pressure at the boundaries
//

__global__ void MixtureBoundaryPressureKernel(int p0_width){
	const int index = blockDim.x * blockIdx.x + threadIdx.x;

	const int i1m = index + array_c.ista1m;
	const int j1m = index + array_c.jsta1m;

	double pamp = mix_params_c.rho_inf * mix_params_c.cs_inf * plane_wave_c.amp;

	double s1, s2;

	if (PML_c.Y0 == 0){
		if((i1m >= array_c.istam) && (i1m <= array_c.iendm)){
			for (int j = array_c.ms; j <= 0; j++){
				mixture_c.p0[p0_width * (j - array_c.jsta1m) + index] = mixture_c.p0[p0_width * (1 - j - array_c.jsta1m) + index];
			}
		}
		__syncthreads();
		if (plane_wave_c.Plane_P == 1){
			if ((i1m >= array_c.istam) && (i1m <= array_c.iendm)){
				if (fmod(tstep_c, ((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq)
					<=
					((double)plane_wave_c.on_wave)/plane_wave_c.freq){
					mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = pamp * sin(plane_wave_c.omega * tstep_c);
				}
				else{
					mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
				}
			}
		}
		__syncthreads();
		if (plane_wave_c.Focused_P == 1){
			if ((i1m >= array_c.istam) && (i1m <= array_c.iendm)){
				double ym = -0.5 * grid_c.dy;
				double xm = (i1m - 0.5) * grid_c.dx;
				s1 = sqrt((plane_wave_c.fp.x - xm) * (plane_wave_c.fp.x - xm) + (plane_wave_c.fp.y - ym) * (plane_wave_c.fp.y - ym));
				if (s1 <= plane_wave_c.f_dist){
					s1 = max(tstep_c - (plane_wave_c.f_dist - s1)/mix_params_c.cs_inf, 0.0);
					s2 = fmod(s1, ((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq);
					if (s2 < (double)plane_wave_c.on_wave/plane_wave_c.freq){
						mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = pamp * sin(plane_wave_c.omega * s2);
					}
					else{
						mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
					}
				}
				else{
					mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
				}
			}
		}
	}
	__syncthreads();
	if (PML_c.Y1 == 0){
		if ((i1m >= array_c.istam) && (i1m <= array_c.iendm)){
			for(int j = grid_c.X; j <= grid_c.X + array_c.me; j++){
				mixture_c.p0[p0_width * (j - array_c.jsta1m) + index] = mixture_c.p0[p0_width * (grid_c.Y - (j - grid_c.Y - 1 - array_c.jsta1m)) + index];
			}
		}
	}
	__syncthreads();
	if (PML_c.X0 == 0){
		if ((j1m >= array_c.jsta1m) && (j1m <= array_c.jend1m)){
			for(int i = array_c.ms; i <= 0; i++){
				mixture_c.p0[p0_width * index + i - array_c.ista1m] = mixture_c.p0[p0_width * index + 1 - i - array_c.ista1m];
			}
		}
	}
	__syncthreads();
	if (PML_c.X1 == 0){
		if ((j1m >= array_c.jsta1m) && (j1m <= array_c.jend1m)){
			for(int i = grid_c.X + 1; i <= grid_c.X + array_c.me; i++){
				mixture_c.p0[p0_width * index + i - array_c.ista1m] = mixture_c.p0[p0_width * index + (grid_c.X - (i - grid_c.X - 1) - array_c.ista1m)];
			}
		}
	}
}

__global__ void MixtureBoundaryPressureKernel_OLD(int p0_width){
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;

	const int i = bx*blockDim.x + tx;
	const int i1m = i + array_c.ista1m;
	const int j1m = i + array_c.jsta1m;

	bool istam_to_iendm = (i1m >= array_c.istam) && (i1m <= array_c.iendm);
	bool jsta1m_to_jend1m = (j1m >= array_c.jsta1m) && (j1m <= array_c.jend1m);

	int p0_i = i;

	double s1, s2;
	double pamp = mix_params_c.rho_inf * mix_params_c.cs_inf * plane_wave_c.amp;
	double p0;

	if((PML_c.Y0 == 0) && istam_to_iendm){
	for (int count = array_c.ms; count <= 0; count++){
		p0 = mixture_c.p0[p0_width * (1 - count - array_c.jsta1m) + i];
		mixture_c.p0[p0_width * (count - array_c.jsta1m) + i] = p0;
	}
	p0_i = p0_width * (0 - array_c.jsta1m) + i;
	if (plane_wave_c.Plane_P == 0){
	if (istam_to_iendm){
	if ((fmod(tstep_c,(double)(plane_wave_c.on_wave + plane_wave_c.off_wave)/plane_wave_c.freq))
		<=
		((double)plane_wave_c.on_wave)/plane_wave_c.freq){
		mixture_c.p0[p0_i] = pamp * sin(plane_wave_c.omega * tstep_c);
	}
	else{
		mixture_c.p0[p0_i] = 0.0;
	}
	}
	}



	if (plane_wave_c.Focused_P == 1){
	if (istam_to_iendm){
		double ym = -0.5 * grid_c.dy;
		double xm = (i1m - 0.5) * grid_c.dx;

		s1 = sqrt(((plane_wave_c.fp.x - xm)*(plane_wave_c.fp.x - xm)) + ((plane_wave_c.fp.y - ym)*(plane_wave_c.fp.y - ym)));
		if (s1 <= plane_wave_c.f_dist){
			s1 = max( tstep_c - ( plane_wave_c.f_dist - s1 ) / mix_params_c.cs_inf, 0.0 );
			s2 = fmod( s1, ((double)(plane_wave_c.on_wave + plane_wave_c.off_wave))/plane_wave_c.freq);
			if (s2 < (double)plane_wave_c.on_wave / plane_wave_c.freq){
				mixture_c.p0[p0_i] = pamp * sin(plane_wave_c.omega * s2);
			}
			else{
				mixture_c.p0[p0_i] = 0.0;
			}
		}
		else{
			mixture_c.p0[p0_i] = 0.0;
		}
	}

	}

	}
	if ((PML_c.Y1 == 0) && istam_to_iendm){
	for (int j = grid_c.Y; j <= grid_c.Y + array_c.me; j++){
		p0_i = p0_width * (grid_c.Y - (j - grid_c.Y - 1) - array_c.jsta1m) + i;
		p0 = mixture_c.p0[p0_i];
		p0_i = p0_width * (j - array_c.jsta1m) + i;
		mixture_c.p0[p0_i] = p0;
	}
	}

	if ((PML_c.X0 == 0) && jsta1m_to_jend1m){
	for (int j = array_c.ms; j <= 0; j++){
		p0_i = p0_width * i + (1 - j) - array_c.ista1m;
		p0 = mixture_c.p0[p0_i];
		p0_i = p0_width * i + j - array_c.ista1m;
		mixture_c.p0[p0_i] = p0;
	}
	}

	if ((PML_c.X1 == 0) && jsta1m_to_jend1m){
	for (int j = grid_c.X + 1; j <= grid_c.X + array_c.me; j++){
		p0_i = p0_width * i + (grid_c.X - (j - grid_c.X - 1)) - array_c.ista1m;
		p0 = mixture_c.p0[p0_i];
		p0_i = p0_width * i + j - array_c.ista1m;
		mixture_c.p0[p0_i] = p0;
	}
	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the conductivity of the mixture based on void fraction
//

__global__ void MixtureKMKernel(int km_width, int T_width, int fg_width){
	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx * blockDim.x + tx;
	const int j = by * blockDim.y + ty;

	const int i2m = i + array_c.ista2m;
	const int j2m = j + array_c.jsta2m;

	const bool ok = ((i2m <= array_c.iend2m) && (j2m <= array_c.jend2m));

	double T, fg;

	if (ok){
		fg = mixture_c.f_g[j * fg_width + i];
		T = mixture_c.T[j * T_width + i];
		mixture_c.k_m[j * km_width + i] = thermal_conductivity_water(T + mix_params_c.T_inf) * (1.0 - 3.0 * fg / (2.0 + fg));
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Zeroes the work array
//

__global__ void WorkClearKernel(int Work_width){
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;

	if ((i + array_c.ista2m <= array_c.iend2m) && (j + array_c.jsta2m <= array_c.jend2m)){
		mixture_c.Work[j*Work_width + i] = 0.0;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates heat transfer from bubbles to mixture
//

__global__ void BubbleHeatKernel(int Work_width){
//	__shared__ double3 Q_list[LINEAR_BLOCK_SIZE][BH_MAX_AREA];
	int2 ibn;
	double2 pos;
	double n_B;
	double Q_B;

	double Q_temp;

	double Delta_x, Delta_y;

	const int bx = blockIdx.x;
	const int tx = threadIdx.x;

	const int index = bx * blockDim.x + tx;

	const bool ok = (index < num_bubbles);

	if (ok){
		ibn = bubbles_c.ibn[index];
		pos = bubbles_c.pos[index];
		n_B = bubbles_c.n_B[index];
		Q_B = bubbles_c.Q_B[index];

	#ifdef _CYLINDRICAL_
		Q_temp = n_B * Q_B * grid_c.rdx * grid_c.rdy / pos.x;
	#else
		Q_temp = n_B * Q_B * grid_c.rdx * grid_c.rdy;
	#endif
	}

//	int count = 0;
	if (ok){
		for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++){
			Delta_x = smooth_delta_x(i, pos.x);
			for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++){
				Delta_y = smooth_delta_y(j, pos.y);
				if((i >= array_c.ista2m) && (i <= array_c.iend2m) && (j >= array_c.jsta2m) && (j <= array_c.jend2m)){
					// Q_list[tx][count] = make_double3(i,j,Q_temp * Delta_x * Delta_y);
					atomicAdd(&mixture_c.Work[(j - array_c.jsta2m) * Work_width + (i - array_c.ista2m)], Q_temp * Delta_x * Delta_y);
//					count++;
				}
			}
		}
	}
	__syncthreads();
/*
	if (tx == 0){
		for (int outer = 0; outer < LINEAR_BLOCK_SIZE; outer ++){
			for (int inner = 0; inner < BH_MAX_AREA; inner ++){
				mixture_c.Work[((int)Q_list[outer][inner].y - array_c.jsta2m) * Work_width + ((int)Q_list[outer][inner].x - array_c.ista2m)] += Q_list[outer][inner].z;
			}
		}
	}
*/
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the mixture temperature
//	Part 1 of a 2 part kernel
//

__global__ void MixtureTemperatureKernel_1(int km_width, int T_width, int Ex_width, int Ey_width){

	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx * blockDim.x + tx;
	const int j = by * blockDim.y + ty;

	const int i2m = i + array_c.ista2m;
	const int j2m = j + array_c.jsta2m;

	double s0, s1;

	__syncthreads();

	if ((i2m >= array_c.ista1n) && (i2m <= array_c.iend1n) && (j2m >= array_c.jsta1m) && (j2m <= array_c.jend1m)){
		s0 = 2.0 * mixture_c.k_m[j * km_width + i] * mixture_c.k_m[j * km_width + i + 1] / (mixture_c.k_m[j * km_width + i] + mixture_c.k_m[j * km_width + i + 1]);
		s1 = ( -mixture_c.T[j * T_width + i] + mixture_c.T[j * T_width + i + 1]) * grid_c.rdx;

		mixture_c.Ex[(j2m - array_c.jsta1m) * Ex_width + i2m - array_c.ista1n] = s0 * s1;
	}

	if ((i2m >= array_c.ista1m) && (i2m <= array_c.iend1m) && (j2m >= array_c.jsta1n) && (j2m <= array_c.jend1n)){
		s0 = 2.0 * mixture_c.k_m[j * km_width + i] * mixture_c.k_m[(j+1) * km_width + i] / (mixture_c.k_m[j * km_width + i] + mixture_c.k_m[(j+1) * km_width + i]);
		s1 = ( -mixture_c.T[j * T_width + i] + mixture_c.T[(j+1) * T_width + i]) * grid_c.rdy;

		mixture_c.Ey[(j2m - array_c.jsta1n) * Ey_width + i2m - array_c.ista1m] = s0 * s1;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the mixture temperature
//	Part 2 of a 2 part kernel
//

__global__ void MixtureTemperatureKernel_2(int T_width, int Ex_width, int Ey_width, int rhom_width, int Cpm_width, int Work_width){
	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx * blockDim.x + tx;
	const int j = by * blockDim.y + ty;

	const int im = i + array_c.ista2m;
	const int jm = j + array_c.jsta2m;

	const bool ok = ((im >= array_c.istam) && (im <= array_c.iendm) && (jm >= array_c.jstam) && (jm <= array_c.jendm));

	double rhom, Cpm, Q;
	double s0, s1, s2;

	if (ok){
		rhom = mixture_c.rho_m[(jm - array_c.jsta1m) * rhom_width + im - array_c.ista1m];
		Cpm = mixture_c.C_pm[(jm - array_c.jsta1m) * Cpm_width + im - array_c.ista1m];
		Q = mixture_c.Work[(jm - array_c.jsta2m) * Work_width + im - array_c.ista2m];
	}

	__syncthreads();

	if (ok){
		s0 = rhom * Cpm;
		s1 = ( -mixture_c.Ex[(jm - array_c.jsta1m) * Ex_width + im - 1 - array_c.ista1n] + mixture_c.Ex[(jm - array_c.jsta1m) * Ex_width + im - array_c.ista1n]) * grid_c.rdx;
		s2 = ( -mixture_c.Ey[(jm - 1 - array_c.jsta1n) * Ey_width + im - array_c.ista1m] + mixture_c.Ey[(jm - array_c.jsta1n) * Ey_width + im - array_c.ista1m]) * grid_c.rdy;
		mixture_c.T[(jm - array_c.jsta2m) * T_width + im - array_c.ista2m] += mix_params_c.dt * (s1 + s2 + Q) / s0;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates the temperature at the boundary of the mixture
//

__global__ void MixtureBoundaryTemperatureKernel(int T_width){
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;

	const int index = bx * blockDim.x + tx;

	const int i2m = index + array_c.ista2m;
	const int j2m = index + array_c.jsta2m;

	if ((PML_c.X0 == 0) && (j2m >= array_c.jstam) && (j2m <= array_c.jendm)){
		for (int i = 1 + array_c.ns + array_c.ms; i <= 0; i++){
			mixture_c.T[index * T_width + i - array_c.ista2m] = mixture_c.T[index * T_width + 1 - i - array_c.ista2m];
		}
	}
	__syncthreads();
	if ((PML_c.X1 == 0) && (j2m >= array_c.jsta1m) && (j2m <= array_c.jend1m)){
		for (int i = grid_c.X + 1; i <= grid_c.X + array_c.ne + array_c.me; i++){
			mixture_c.T[index * T_width + i - array_c.ista2m] = mixture_c.T[index * T_width + grid_c.X - (i - grid_c.X - 1) - array_c.ista2m];
		}
	}
	__syncthreads();
	if ((PML_c.Y0 == 0) && (i2m <= array_c.iend2m)){
		for (int j = 1 + array_c.ns + array_c.ms; j <= 0; j++){
			mixture_c.T[(j - array_c.jsta2m) * T_width + index] = mixture_c.T[(1 - j - array_c.jsta2m) * T_width + index];
		}
	}
	__syncthreads();
	if ((PML_c.Y1 == 0) && (i2m <= array_c.iend2m)){
		for (int j = grid_c.Y; j <= grid_c.Y + array_c.ne + array_c.me; j++){
			mixture_c.T[(j - array_c.jsta2m) * T_width + index] = mixture_c.T[(grid_c.Y - (j - grid_c.Y - 1) - array_c.jsta2m) * T_width + index];
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//	Calculates various mixture properties:
//	Density (liquid and mixture)
//	Speed of sound
//	Specific heat capacity
//

__global__ void MixturePropertiesKernel(int rhol_width, int rhom_width, int csl_width, int cpm_width, int fg_width, int T_width){
	const int bx = blockIdx.x,	by = blockIdx.y;
	const int tx = threadIdx.x,	ty = threadIdx.y;

	const int i = bx * blockDim.x + tx;
	const int j = by * blockDim.y + ty;

	const int i1m = i + array_c.ista1m;
	const int j1m = j + array_c.jsta1m;

	if ((i1m <= array_c.iend1m) && (j1m <= array_c.jend1m)){
		mixture_c.rho_l[j * rhol_width + i] = mix_params_c.rho_inf;
		mixture_c.c_sl[j * csl_width + i] = mix_params_c.cs_inf;
		mixture_c.rho_m[j * rhom_width + i] = (1.0 - mixture_c.f_g[(j1m - array_c.jsta2m) * fg_width + i1m - array_c.ista2m]) * mix_params_c.rho_inf;
		mixture_c.C_pm[j * cpm_width + i] = specific_heat_water(mixture_c.T[(j1m - array_c.jsta2m) * T_width + i1m - array_c.ista2m] + mix_params_c.T_inf);
	}
}

/**************************************************************************************************
 *                                        Device Functions                                        *
 **************************************************************************************************/


///////////////////////////////////////////////////////////////////////////////////////////////////
// Physical Properties of Water
//

__inline__ __host__ __device__ double density_water(double P, double T){
	return 1000.0;
} // water_density()

__inline__ __host__ __device__ double thermal_expansion_rate_water(double P, double T){
	return 200.0e-6f;
} // thermal_expansion_rate_water()

__inline__ __host__ __device__ double adiabatic_sound_speed_water(double P, double T){
	return 1500.0;
} // adiabatic_sound_speed_water()

__inline__ __host__ __device__ double viscosity_water(double T){
	return exp(-10.4349f - 507.881f/(149.390-T));
} // viscosity_water()

__inline__ __host__ __device__ double thermal_conductivity_water(double T){
	return 0.76760 + (7.5390e-3f * T) - 9.8250e-6f * (T*T);
} // thermal_conductivity()

__inline__ __host__ __device__ double specific_heat_water(double T){
	double H2O = 18.0e-3f;
	return  (917.5 - 10.1016f * T + 0.0454134f * (T*T) - 9.07517e-5 * (T*T*T) + 6.80700e-8f * (T*T*T*T))/H2O;
} // specific_heat_water()

__inline__ __host__ __device__ double vapor_pressure_water(double T){
	double Tc = T - 273.15;
	return 6.11176750
		+ 0.443986062f * Tc
		+ 0.143053301e-1f * Tc*Tc
		+ 0.265027242e-3f * pow(Tc,3)
		+ 0.302246994e-5 * pow(Tc,4)
		+ 0.203886313e-7f * pow(Tc,5)
		+ 0.638780966e-10 * pow(Tc,6);
} // vapor_pressure_water()

__inline__ __host__ __device__ double surface_tension_water(double T){
	return 0.12196f - 0.1676e-3f * T;
} // surface_tension_water()

///////////////////////////////////////////////////////////////////////////////////////////////////
// Smooth Delta Function
//

// Void Fraction from Bubble
__inline__ __host__ __device__ double fg_from_bubble(	const int2 	pos,
					const double2 	bubPos,
					const double 	n_b,
					const double 	R_t){
#ifdef _CYLINDRICAL_
	return Pi4r3 * n_b * (R_t * R_t * R_t) * grid_c.rdx * grid_c.rdy / bubPos.x
		* smooth_delta_x(pos.x, bubPos.x)
		* smooth_delta_y(pos.y, bubPos.y);
#else
	return Pi4r3 * n_b * (R_t * R_t * R_t) * grid_c.rdx * grid_c.rdy
		* smooth_delta_x(pos.x, bubPos.x)
		* smooth_delta_y(pos.y, bubPos.y);
#endif
} // fg_from_bubble()

__inline__ __host__ __device__ double smooth_delta_x(	const int 	pos,
					const double 	bub_x){

	return (1.0/((double)sim_params_c.deltaBand))
		* (1.0 + cos((2.0 * Pi / ((double)sim_params_c.deltaBand)
		* grid_c.rdx) * (((double)pos - 0.5)*grid_c.dx - bub_x)));
} // smooth_delta_x()

__inline__ __host__ __device__ double smooth_delta_y(	const int 	pos,
					const double 	bub_y){

	return (1.0/((double)sim_params_c.deltaBand))
		* (1.0 + cos((2.0 * Pi / ((double)sim_params_c.deltaBand)
		* grid_c.rdy) * (((double)pos - 0.5)*grid_c.dy - bub_y)));
} // smooth_delta_y()




///////////////////////////////////////////////////////////////////////////////////////////////////
//	Wrapper Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

int update_bubble_indices(){
	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	BubbleUpdateIndexKernel <<< dimBubbleGrid, dimBubbleBlock >>> ();
	cudaThreadSynchronize();
	checkCUDAError("Bubble Update Index");

	return 0;
}

int calculate_void_fraction(mixture_t mixture_htod, int f_g_width, int f_g_pitch){

	dim3 dim2mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dim2mGrid((i2m + TILE_BLOCK_WIDTH - 1) / TILE_BLOCK_WIDTH,
			(j2m + TILE_BLOCK_HEIGHT - 1) / TILE_BLOCK_HEIGHT);

	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

//	VoidFractionClearKernel <<< dim2mGrid, dim2mBlock >>> (f_g_width);
//	cudaThreadSynchronize();
//	checkCUDAError("Void Fraction Clear");

	cudaMemset2D(mixture_htod.f_g, f_g_pitch, 0, i2m * sizeof(double), j2m);
	cudaThreadSynchronize();
	checkCUDAError("Clear Void Fraction");

	VoidFractionReverseLookupKernel <<< dimBubbleGrid, dimBubbleBlock >>> (f_g_width);
	cudaThreadSynchronize();
	checkCUDAError("Void Fraction Lookup");

	VoidFractionCylinderKernel <<< dim2mGrid, dim2mBlock >>> (f_g_width);
	cudaThreadSynchronize();
	checkCUDAError("Void Fraction Cylindrical Conditions");

	return 0;
}

int synchronize_void_fraction(mixture_t mixture_htod, size_t f_g_pitch){

	cudaMemcpy2D(mixture_htod.f_gn, f_g_pitch, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(mixture_htod.f_gm, f_g_pitch, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice);

	return 0;
}

int store_variables(mixture_t mixture_htod, bubble_t bubbles_htod, int f_g_width, int p_width, int Work_width, size_t f_g_pitch, size_t Work_pitch, size_t p_pitch){
	dim3 dim2mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dim2mGrid((i2m + TILE_BLOCK_WIDTH - 1) / TILE_BLOCK_WIDTH,
			(j2m + TILE_BLOCK_HEIGHT - 1) / TILE_BLOCK_HEIGHT);

	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	WorkClearKernel <<< dim2mGrid, dim2mBlock >>> (Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Clear Work Kernel");

	// Void fraction (fg) prediction, store it in the work array
	VFPredictionKernel <<< dim2mGrid, dim2mBlock >>> (f_g_width);
	cudaThreadSynchronize();
	checkCUDAError("Void Fraction Prediction");

	// Store variables
//	MixStoreKernel <<< dim2mGrid, dim2mBlock >>> (f_g_width, p_width);
//	cudaThreadSynchronize();
//	checkCUDAError("Store Mixture");


	cudaMemcpy2D(	mixture_htod.pn, p_pitch,
			mixture_htod.p, p_pitch,
			sizeof(double2)*i1m, j1m,
			cudaMemcpyDeviceToDevice);

	cudaMemcpy2D(	mixture_htod.f_gm, f_g_pitch,
			mixture_htod.f_gn, f_g_pitch,
			sizeof(double)*i2m, j2m,
			cudaMemcpyDeviceToDevice);

	cudaMemcpy2D(	mixture_htod.f_gn, f_g_pitch,
			mixture_htod.f_g, f_g_pitch,
			sizeof(double)*i2m, j2m,
			cudaMemcpyDeviceToDevice);


//	BubStoreKernel <<< dimBubbleGrid, dimBubbleBlock >>> ();
//	cudaThreadSynchronize();
//	checkCUDAError("Store Bubbles");

	cudaMemcpy(bubbles_htod.R_pn, bubbles_htod.R_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.R_nn, bubbles_htod.R_n, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.d1_R_n, bubbles_htod.d1_R_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.PG_n, bubbles_htod.PG_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.PL_m, bubbles_htod.PL_n, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.PL_n, bubbles_htod.PL_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.dt_n, bubbles_htod.dt, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);
	cudaMemcpy(bubbles_htod.re_n, bubbles_htod.re, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice);

	// Store the predicted void fractions (fg)
	cudaMemcpy2D(	mixture_htod.f_g, f_g_pitch,
			mixture_htod.Work, Work_pitch,
			sizeof(double)*i2m, j2m,
			cudaMemcpyDeviceToDevice);

	checkCUDAError("Store Predicted Void Fraction");

	return 0;
}

int calculate_velocity_field(int vx_width, int vy_width, int rho_m_width, int p0_width, int c_sl_width){
	dim3 dimVXBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dimVXGrid((i2n + TILE_BLOCK_WIDTH - 1) / (TILE_BLOCK_WIDTH),
			(j2m + TILE_BLOCK_HEIGHT - 1) / (TILE_BLOCK_HEIGHT));

	dim3 dimVXBBlock(LINEAR_BLOCK_SIZE);
	dim3 dimVXBGrid((max(i2n,j2m) + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	dim3 dimVYBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dimVYGrid((i2m + TILE_BLOCK_WIDTH - 1) / (TILE_BLOCK_WIDTH),
			(j2n + TILE_BLOCK_HEIGHT - 1) / (TILE_BLOCK_HEIGHT));

	dim3 dimVYBBlock(LINEAR_BLOCK_SIZE);
	dim3 dimVYBGrid((max(i2m,j2n) + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));
	
	dim3 dimVelocityBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dimVelocityGrid((max(i2m, i2n) + TILE_BLOCK_WIDTH - 1) / (TILE_BLOCK_WIDTH),
			(max(j2m, j2n) + TILE_BLOCK_HEIGHT - 1) / (TILE_BLOCK_HEIGHT));

//	cudaStream_t vx_stream, vy_stream;
//	cudaStreamCreate(&vx_stream);
//	cudaStreamCreate(&vy_stream);

	VelocityKernel <<< dimVelocityGrid, dimVelocityBlock, 3 * (TILE_BLOCK_WIDTH + 1) * (TILE_BLOCK_HEIGHT + 1) * sizeof(double) >>> (vx_width, vy_width, rho_m_width, p0_width, c_sl_width);

//	VXKernel <<< dimVXGrid, dimVXBlock, 3 * TILE_BLOCK_WIDTH * (TILE_BLOCK_HEIGHT + 1) * sizeof(double), vx_stream>>> (vx_width, rho_m_width, p0_width, c_sl_width);
//	//cudaThreadSynchronize();
//	checkCUDAError("VX");
//	VYKernel <<< dimVYGrid, dimVYBlock, 3 * TILE_BLOCK_WIDTH * (TILE_BLOCK_HEIGHT + 1) * sizeof(double), vy_stream >>> (vy_width, rho_m_width, p0_width, c_sl_width);
//	//cudaThreadSynchronize();
//	checkCUDAError("VY");

//	VXBoundaryKernel <<< dimVXBGrid, dimVXBBlock, 0, vx_stream >>> (vx_width);
//	//cudaThreadSynchronize();
//	checkCUDAError("VX Boundary");
//	VYBoundaryKernel <<< dimVYBGrid, dimVYBBlock, 0, vy_stream >>> (vy_width);
//	//cudaThreadSynchronize();
//	checkCUDAError("VY Boundary");

	VXBoundaryKernel <<< dimVXBGrid, dimVXBBlock >>> (vx_width);
	cudaThreadSynchronize();
	checkCUDAError("VX Boundary");
	VYBoundaryKernel <<< dimVYBGrid, dimVYBBlock >>> (vy_width);
	cudaThreadSynchronize();
	checkCUDAError("VY Boundary");

//	cudaStreamDestroy(vx_stream);
//	cudaStreamDestroy(vy_stream);

	return 0;
}

int bubble_motion(bubble_t bubbles_htod, int vx_width, int vy_width){
	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

//	dim3 dimBIVBlock(HALF_LINEAR_SIZE, DIMENSION);
//	dim3 dimBIVGrid((numBubbles + HALF_LINEAR_SIZE - 1) / (HALF_LINEAR_SIZE));

	dim3 dimBMBlock(HALF_LINEAR_SIZE, DIMENSION);
	dim3 dimBMGrid((numBubbles + HALF_LINEAR_SIZE - 1) / (HALF_LINEAR_SIZE));

	// Calculate each bubble's interpolated liquid velocity
//	BubbleInterpolationVelocityKernel <<< dimBIVGrid, dimBIVBlock >>> (vx_width, vy_width);
	BubbleInterpolationVelocityKernel <<< dimBubbleGrid, dimBubbleBlock >>> (vx_width, vy_width);
	cudaThreadSynchronize();
	checkCUDAError("Bubble Interpolation Velocity");

	// Calculate (read: copy) the bubble velocity from liquid velocity
	cudaMemcpy(	bubbles_htod.v_B,
			bubbles_htod.v_L,
			sizeof(double2)*numBubbles,
			cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();
	checkCUDAError("Bubble Velocity");

	// Move Bubbles
	BubbleMotionKernel <<< dimBMGrid, dimBMBlock >>> ();
	cudaThreadSynchronize();
	checkCUDAError("Bubble Motion");

	BubbleUpdateIndexKernel <<< dimBubbleGrid, dimBubbleBlock >>> ();
	cudaThreadSynchronize();
	checkCUDAError("Bubble Update Index");

	return 0;
}

double calculate_pressure_field(mixture_t mixture_h, mixture_t mixture_htod, double P_inf, int p0_width, int p_width, int f_g_width, int vx_width, int vy_width, int rho_l_width, int c_sl_width, int Work_width, size_t Work_pitch){
	double resimax = 0.0;

	dim3 dimMBPBlock(LINEAR_BLOCK_SIZE);
	dim3 dimMBPGrid((max(i1m, j1m) + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	dim3 dim1mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dim1mGrid((i1m + TILE_BLOCK_WIDTH - 1)/TILE_BLOCK_WIDTH,
			(j1m + TILE_BLOCK_HEIGHT - 1)/TILE_BLOCK_HEIGHT);

	WorkClearKernel <<< dim1mGrid, dim1mBlock >>> (Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Clear Work Kernel");

	MixturePressureKernel <<< dim1mGrid, dim1mBlock , 3 * (TILE_BLOCK_WIDTH + 1) * (TILE_BLOCK_HEIGHT + 1) * sizeof(double) >>> (vx_width, vy_width, f_g_width, rho_l_width, c_sl_width, p0_width, p_width, Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture Pressure");

	cudaMemcpy2D(	mixture_h.Work, sizeof(double)*i2m,
			mixture_htod.Work, Work_pitch,
			sizeof(double)*i2m, j2m,
			cudaMemcpyDeviceToHost);
	checkCUDAError("Work Array Copy");

	MixtureBoundaryPressureKernel <<< dimMBPGrid, dimMBPBlock >>> (p0_width);

	resimax = 0;
	for (int i = 0; i < i2m * j2m; i++){
		if (resimax < mixture_h.Work[i]){
		resimax = mixture_h.Work[i];
		}
	}
	resimax /= P_inf;
	cudaThreadSynchronize();
	checkCUDAError("Boundary Pressure");

	return resimax;
}

int interpolate_bubble_pressure(int p0_width){
	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	BubbleInterpolationScalarKernel <<< dimBubbleGrid, dimBubbleBlock >>> (p0_width);
	cudaThreadSynchronize();
	checkCUDAError("Bubble interpolation scalar");

	return 0;
}

int calculate_temperature(int k_m_width, int T_width, int f_g_width, int Ex_width, int Ey_width, int rho_m_width, int C_pm_width, int Work_width){
	dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
	dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

	dim3 dim2mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dim2mGrid((i2m + TILE_BLOCK_WIDTH - 1) / TILE_BLOCK_WIDTH,
			(j2m + TILE_BLOCK_HEIGHT - 1) / TILE_BLOCK_HEIGHT);

	dim3 dimMBTBlock(LINEAR_BLOCK_SIZE);
	dim3 dimMBTGrid((max(i2m, j2m) + LINEAR_BLOCK_SIZE - 1)/LINEAR_BLOCK_SIZE);


	MixtureKMKernel <<< dim2mGrid, dim2mBlock >>> (k_m_width, T_width, f_g_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture KM");

	WorkClearKernel <<< dim2mGrid, dim2mBlock >>> (Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Clear Work Kernel");

	BubbleHeatKernel <<< dimBubbleGrid, dimBubbleBlock >>> (Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Bubble heat");

	MixtureTemperatureKernel_1 <<< dim2mGrid, dim2mBlock >>> (k_m_width, T_width, Ex_width, Ey_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture temperature 1");

	MixtureTemperatureKernel_2 <<< dim2mGrid, dim2mBlock >>> (T_width, Ex_width, Ey_width, rho_m_width, C_pm_width, Work_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture temperature 1");

	MixtureBoundaryTemperatureKernel <<< dimMBTGrid, dimMBTBlock >>> (T_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture boundary temperature");

	return 0;
}

int calculate_properties(int rho_l_width, int rho_m_width, int c_sl_width, int C_pm_width, int f_g_width, int T_width){
	dim3 dim1mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
	dim3 dim1mGrid((i1m + TILE_BLOCK_WIDTH - 1)/TILE_BLOCK_WIDTH,
			(j1m + TILE_BLOCK_HEIGHT - 1)/TILE_BLOCK_HEIGHT);

	MixturePropertiesKernel <<< dim1mGrid, dim1mBlock >>> (rho_l_width, rho_m_width, c_sl_width, C_pm_width, f_g_width, T_width);
	cudaThreadSynchronize();
	checkCUDAError("Mixture properties");

	return 0;
}

// call Bubbles_Radius
//     $( Rt_b(n), Rp_b(n), Rn_b(n), Rpn_b(n), Rnn_b(n)
//     $, d1Rp_b(n), d1Rn_b(n), PGp_b(n), PGn_b(n)
//     $, PLp_b(n), PLn_b(n), PLm_b(n)
//     $, QB_b(n), dt_b(n), dtn_b(n), dt0_b, dt, re_b(n), ren_b(n) )

int solve_bubble_radii(bubble_t bubbles_htod){

	const int block = 512;
	dim3 dimBubbleBlock(block);
	dim3 dimBubbleGrid((numBubbles / 1 + block - 1) / (block));

	int * max_iter_d, max_iter_h[numBubbles];
	int findmaxiter = 0;

	cudaMalloc((void**)&max_iter_d, sizeof(int) * numBubbles);

	BubbleRadiusKernel <<< dimBubbleGrid, dimBubbleBlock>>> (max_iter_d);
	cudaThreadSynchronize();
	checkCUDAError("Bubble Radius");

	cudaMemcpy(max_iter_h, max_iter_d, sizeof(int)*numBubbles, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numBubbles; i++){
		if (max_iter_h[i] > findmaxiter) findmaxiter = max_iter_h[i];
	}

	cutilSafeCall(cudaFree(max_iter_d));

	return findmaxiter;
}

int solve_bubble_radii_host(bubble_t bubbles_h, bubble_t bubbles_htod, bub_params_t bub_params, mix_params_t mix_params){
	double Rp, Rn, d1Rp, PGp, dt_L, remain;
	double PC0, PC1, PC2, thetime;

	int debugcount;

	//double d1Rn
	double PGn, PL, Rt, omega_N;
	double dTdr_R, SumHeat, SumVis;
	doublecomplex alpha_N, Lp_N;

	int maxiter = 0;

	cudaMemcpy(bubbles_h.ibm,	bubbles_htod.ibm,		sizeof(int2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.ibn,	bubbles_htod.ibn,		sizeof(int2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.pos,	bubbles_htod.pos,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.R_t,	bubbles_htod.R_t,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.R_p,	bubbles_htod.R_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.R_pn,	bubbles_htod.R_pn,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.R_n,	bubbles_htod.R_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.R_nn,	bubbles_htod.R_nn,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.d1_R_p,	bubbles_htod.d1_R_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.d1_R_n,	bubbles_htod.d1_R_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.PG_p,	bubbles_htod.PG_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.PG_n,	bubbles_htod.PG_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.PL_p,	bubbles_htod.PL_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.PL_n,	bubbles_htod.PL_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.PL_m,	bubbles_htod.PL_m,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.Q_B,	bubbles_htod.Q_B,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.n_B,	bubbles_htod.n_B,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.dt,	bubbles_htod.dt,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.dt_n,	bubbles_htod.dt_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.re,	bubbles_htod.re,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.re_n,	bubbles_htod.re_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.v_B,	bubbles_htod.v_B,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(bubbles_h.v_L,	bubbles_htod.v_L,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);

	int max_iter_device = solve_bubble_radii(bubbles_htod);

	for (int index = 0; index < numBubbles; index++){
		//for (index; (index < index + blockDim.x) && (index < num_bubbles); index++){

		Rp 	= bubbles_h.R_pn[index];
		Rn 	= bubbles_h.R_nn[index];
		d1Rp 	= bubbles_h.d1_R_n[index];
		PGp	= bubbles_h.PG_n[index];
		dt_L	= bubbles_h.dt_n[index];
		remain	= bubbles_h.re_n[index] + mix_params.dt;

		PC0 = bubbles_h.PL_n[index] + bub_params.PL0;
		PC1 = 0.5*(bubbles_h.PL_p[index]-bubbles_h.PL_m[index])/mix_params.dt;
		PC2 = 0.5*(bubbles_h.PL_p[index]+bubbles_h.PL_m[index]-2.0*bubbles_h.PL_n[index])/(mix_params.dt * mix_params.dt);

		thetime = -bubbles_h.re_n[index];

		debugcount = 0;
		while (remain > 0.0){
			debugcount ++;
			//d1Rn 	= 	d1Rp;
			PGn 	= 	PGp;
			PL 	= 	PC2 * thetime * thetime + PC1 * thetime + PC0;

			solveRayleighPlesset(&Rt, &Rp, &Rn, &d1Rp, &PGn, &PL, &dt_L, &remain, bub_params);
			thetime = thetime + dt_L;

			omega_N = solveOmegaN (&alpha_N, PGn, Rn, bub_params);

			Lp_N = solveLp(alpha_N, Rn);

			PGp = solvePG(PGn, Rp, Rn, omega_N, dt_L, Lp_N, bub_params);

			dTdr_R 	= 	Lp_N.real() / (Lp_N.abs() * Lp_N.abs()) * bub_params.T0 / (bub_params.PG0 * bub_params.R03) *
					(bub_params.PG0 * bub_params.R03 - 0.5*(PGp+PGn)*(0.5*(Rp+Rn))*(0.5*(Rp+Rn))*(0.5*(Rp+Rn))) +
					Lp_N.imag() / (Lp_N.abs() * Lp_N.abs()) * bub_params.T0 / (bub_params.PG0 * bub_params.R03) *
					(PGp * Rp * Rp * Rp - PGn * Rn * Rn * Rn) / (omega_N * dt_L);

			SumHeat -= 4.0 * Pi * Rp * Rp * bub_params.K0 * dTdr_R * dt_L;
			SumVis 	+= 4.0 * Pi * Rp * Rp * 4.0 * bub_params.mu * d1Rp / Rp * d1Rp * dt_L;

		}
		if (debugcount > maxiter) {maxiter = debugcount;}

		// Assign values back to the global memory
		bubbles_h.R_t[index] 	= Rt;
		bubbles_h.R_p[index] 	= Rp;
		bubbles_h.R_n[index] 	= Rn;
		bubbles_h.d1_R_p[index]	= d1Rp;
		bubbles_h.PG_p[index] 	= PGp;
		bubbles_h.dt[index]	= dt_L;
		bubbles_h.re[index]	= remain;
		bubbles_h.Q_B[index]	= (SumHeat + SumVis) / (mix_params.dt - remain + bubbles_h.re_n[index]);

		//}
	}


	bubble_t derp;

	derp.ibm	= (int2*) calloc(numBubbles,  sizeof(int2));
	derp.ibn	= (int2*) calloc(numBubbles,  sizeof(int2));
	derp.pos	= (double2*) calloc(numBubbles,  sizeof(double2));
	derp.R_t	= (double*) calloc(numBubbles,  sizeof(double));
	derp.R_p	= (double*) calloc(numBubbles,  sizeof(double));
	derp.R_pn	= (double*) calloc(numBubbles,  sizeof(double));
	derp.R_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.R_nn	= (double*) calloc(numBubbles,  sizeof(double));
	derp.d1_R_p	= (double*) calloc(numBubbles,  sizeof(double));
	derp.d1_R_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.PG_p	= (double*) calloc(numBubbles,  sizeof(double));
	derp.PG_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.PL_p	= (double*) calloc(numBubbles,  sizeof(double));
	derp.PL_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.PL_m	= (double*) calloc(numBubbles,  sizeof(double));
	derp.Q_B	= (double*) calloc(numBubbles,  sizeof(double));
	derp.n_B	= (double*) calloc(numBubbles,  sizeof(double));
	derp.dt	= (double*) calloc(numBubbles,  sizeof(double));
	derp.dt_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.re	= (double*) calloc(numBubbles,  sizeof(double));
	derp.re_n	= (double*) calloc(numBubbles,  sizeof(double));
	derp.v_B	= (double2*) calloc(numBubbles,  sizeof(double2));
	derp.v_L	= (double2*) calloc(numBubbles,  sizeof(double2));



	cudaMemcpy(derp.ibm,	bubbles_htod.ibm,		sizeof(int2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.ibn,	bubbles_htod.ibn,		sizeof(int2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.pos,	bubbles_htod.pos,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.R_t,	bubbles_htod.R_t,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.R_p,	bubbles_htod.R_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.R_pn,	bubbles_htod.R_pn,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.R_n,	bubbles_htod.R_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.R_nn,	bubbles_htod.R_nn,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.d1_R_p,	bubbles_htod.d1_R_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.d1_R_n,	bubbles_htod.d1_R_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.PG_p,	bubbles_htod.PG_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.PG_n,	bubbles_htod.PG_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.PL_p,	bubbles_htod.PL_p,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.PL_n,	bubbles_htod.PL_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.PL_m,	bubbles_htod.PL_m,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.Q_B,	bubbles_htod.Q_B,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.n_B,	bubbles_htod.n_B,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.dt,	bubbles_htod.dt,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.dt_n,	bubbles_htod.dt_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.re,	bubbles_htod.re,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.re_n,	bubbles_htod.re_n,		sizeof(double)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.v_B,	bubbles_htod.v_B,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(derp.v_L,	bubbles_htod.v_L,		sizeof(double2)*numBubbles,	cudaMemcpyDeviceToHost);


	double delta[8] = {0,0,0,0,0,0,0,0};

	for (int i = 0; i < numBubbles; i++){
		delta[0]+=(bubbles_h.R_t[i] - derp.R_t[i]);
		delta[1]+=(bubbles_h.R_p[i] - derp.R_p[i]);
		delta[2]+=(bubbles_h.R_n[i] - derp.R_n[i]);
		delta[3]+=(bubbles_h.d1_R_p[i] - derp.d1_R_p[i]);
		delta[4]+=(bubbles_h.PG_p[i] - derp.PG_p[i]);
		delta[5]+=(bubbles_h.dt[i] - derp.dt[i]);
		delta[6]+=(bubbles_h.Q_B[i] - derp.Q_B[i]);
		delta[7]+=(bubbles_h.re[i] - derp.re[i]);

	}
	if (delta[0])printf("Delta R_t = %4.2E  \n", delta[0] / numBubbles);
	if (delta[1])printf("Delta R_p = %4.2E  \n", delta[1] / numBubbles);
	if (delta[2])printf("Delta R_n = %4.2E  \n", delta[2] / numBubbles);
	if (delta[3])printf("Delta d1_R_p = %4.2E  \n", delta[3] / numBubbles);
	if (delta[4])printf("Delta PG_p = %4.2E  \n", delta[4] / numBubbles);
	if (delta[5])printf("Delta dt = %4.2E  \n", delta[5] / numBubbles);
	if (delta[6])printf("Delta Q_B = %4.2E  \n", delta[6] / numBubbles);
	if (delta[7])printf("Delta re = %4.2E\n", delta[7] / numBubbles);
	printf("\n");


	cudaMemcpy(bubbles_htod.ibm,	bubbles_h.ibm,		sizeof(int2)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.ibn,	bubbles_h.ibn,		sizeof(int2)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.pos,	bubbles_h.pos,		sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.R_t,	bubbles_h.R_t,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.R_p,	bubbles_h.R_p,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.R_pn,	bubbles_h.R_pn,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.R_n,	bubbles_h.R_n,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.R_nn,	bubbles_h.R_nn,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.d1_R_p,	bubbles_h.d1_R_p,	sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.d1_R_n,	bubbles_h.d1_R_n,	sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.PG_p,	bubbles_h.PG_p,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.PG_n,	bubbles_h.PG_n,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.PL_p,	bubbles_h.PL_p,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.PL_n,	bubbles_h.PL_n,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.PL_m,	bubbles_h.PL_m,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.Q_B,	bubbles_h.Q_B,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.n_B,	bubbles_h.n_B,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.dt,	bubbles_h.dt,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.dt_n,	bubbles_h.dt_n,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.re,	bubbles_h.re,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.re_n,	bubbles_h.re_n,		sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.v_B,	bubbles_h.v_B,		sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
	cudaMemcpy(bubbles_htod.v_L,	bubbles_h.v_L,		sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
	printf("The device iterated %i times\n", max_iter_device);
	return maxiter;
}

