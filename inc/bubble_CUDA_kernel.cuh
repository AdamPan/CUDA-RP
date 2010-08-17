#include "bubble_CUDA.h"
#include "complex.cuh"
#include "double_vector_math.cuh"

#define BUB_RAD_MAX_THREADS 256
#define BUB_RAD_MIN_BLOCKS 2

// Array sizes
extern int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;
extern int numBubbles;

/* Constant Memory */
__device__ __constant__	double		Pi;
__device__ __constant__	double		Pi4r3;

__device__ __constant__	mixture_t	mixture_c;	// Pointers to bubble variables
__device__ __constant__	bubble_t	bubbles_c;	// Pointers to mixture variables

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
__device__ __constant__	double		tstep_c;	// Current time
__device__ __constant__ double3		delta_coef;	// Coefficients used to save ops in calculating the smooth delta function


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
static __inline__ __host__ __device__ double smooth_delta_x(const int, const double);	// x-direction (includes cylindrical case)
static __inline__ __host__ __device__ double smooth_delta_y(const int, const double);	// y-direction

// Functions used by the bubble radius solver
static __forceinline__ __host__ __device__ void solveRayleighPlesset(double * , double * , double * , double * , const double, const double, double * , double *, const bub_params_t);	// Solves the Rayleigh Plesset equation
static __forceinline__ __host__ __device__ double solveOmegaN(doublecomplex *, const double, const double, const bub_params_t);	// Solves for omega_N
static __forceinline__ __host__ __device__ doublecomplex Alpha(const double, const double, const double);	// Calculates alpha_N
static __forceinline__ __host__ __device__ doublecomplex Upsilon(const doublecomplex , const double);	// Calculates Upsilon
static __forceinline__ __host__ __device__ doublecomplex solveLp(const doublecomplex, const double);	// Calculates Lp_N
static __forceinline__ __host__ __device__ double solvePG(const double, const double, const double, const double, const double, const doublecomplex, const bub_params_t);	// Calculates PG

/* Static utility functions */

// Sorts bubbles
static __host__ void sort(bubble_t, thrust::device_vector<int>);

// Implements atomic addition for doubles
static __inline__ __device__ double atomicAdd(double * addr, double val)
{
    double old = *addr;
    double assumed;

    do
    {
        assumed = old;
        old = __longlong_as_double(	atomicCAS((unsigned long long int*)addr,
                                              __double_as_longlong(assumed),
                                              __double_as_longlong(val+assumed)));	// Pull down down the variable, and compare and swap when we have the chance. Note that atomicCAS returns the new value at addr.
    }
    while ( assumed!=old );

    return old;	// We return the old value, to conform with other atomicAdd() implementations
}

// intrinsic epsilon for double
static __inline__ __host__ __device__ double epsilon(double val)
{
    return 2.22044604925031308e-016;
}

// Physical Properties of Water

__inline__ __host__ __device__ double density_water(double P, double T)
{
    return 1000.0;
} // water_density()

__inline__ __host__ __device__ double thermal_expansion_rate_water(double P, double T)
{
    return 200.0e-6f;
} // thermal_expansion_rate_water()

__inline__ __host__ __device__ double adiabatic_sound_speed_water(double P, double T)
{
    return 1500.0;
} // adiabatic_sound_speed_water()

__inline__ __host__ __device__ double viscosity_water(double T)
{
    return exp(-10.4349f - 507.881f/(149.390-T));
} // viscosity_water()

__inline__ __host__ __device__ double thermal_conductivity_water(double T)
{
    return 0.76760 + (7.5390e-3f * T) - 9.8250e-6f * (T*T);
} // thermal_conductivity()

__inline__ __host__ __device__ double specific_heat_water(double T)
{
    double H2O = 18.0e-3f;
    return  (917.5 - 10.1016f * T + 0.0454134f * (T*T) - 9.07517e-5 * (T*T*T) + 6.80700e-8f * (T*T*T*T))/H2O;
} // specific_heat_water()

__inline__ __host__ __device__ double vapor_pressure_water(double T)
{
    double Tc = T - 273.15;
    return 6.11176750
           + 0.443986062f * Tc
           + 0.143053301e-1f * Tc*Tc
           + 0.265027242e-3f * pow(Tc,3)
           + 0.302246994e-5 * pow(Tc,4)
           + 0.203886313e-7f * pow(Tc,5)
           + 0.638780966e-10 * pow(Tc,6);
} // vapor_pressure_water()

__inline__ __host__ __device__ double surface_tension_water(double T)
{
    return 0.12196f - 0.1676e-3f * T;
} // surface_tension_water()

// Smooth Delta Functions
__inline__ __host__ __device__ double smooth_delta_x(	const int pos,
        const double bub_x)
{
//	return delta_coef.x * (1.0 + cos(delta_coef.y * (((double)pos - 0.5)*grid_c.dx - bub_x)));
    return (1.0/((double)sim_params_c.deltaBand)) * (1.0 + cos((2.0 * Pi / ((double)sim_params_c.deltaBand) * grid_c.rdx) * (((double)pos - 0.5)*grid_c.dx - bub_x)));
} // smooth_delta_x()
__inline__ __host__ __device__ double smooth_delta_y(	const int pos,
        const double bub_y)
{
//	return delta_coef.x * (1.0 + cos(delta_coef.z * (((double)pos - 0.5)*grid_c.dy - bub_y)));
    return (1.0/((double)sim_params_c.deltaBand)) * (1.0 + cos((2.0 * Pi / ((double)sim_params_c.deltaBand) * grid_c.rdy) * (((double)pos - 0.5)*grid_c.dy - bub_y)));
} // smooth_delta_y()

/* Reduced Order Bubble Dynamics Modelling Functions */

// Rayleigh Plesset solver
__forceinline__ __host__ __device__ void solveRayleighPlesset(double * Rt, double *Rp, double * Rn, double * d1R, const double PG, const double PL, double * dt, double * remain, const bub_params_t bub_params)
{
    double 	Rm, Rnd2R, 	// Temporary radius variables
    dtm, dtp; 	// Temporary time variables
    // Step the radius forwards
    Rm = *Rn;
    *Rn = *Rp;
    // Calculate the effect of pressure on the
    Rnd2R = (PG - PL - 2.0*bub_params.sig/(*Rn) - 4.0*bub_params.mu/(*Rn)*(*d1R) )/bub_params.rho - 1.5*(*d1R)*(*d1R);
    // Set dt based on either the first derivative of radius, or the second derivative
    dtm = *dt;
    dtp = min(1.01 * (*dt), bub_params.dt0);
    if (abs(*d1R) > epsilon(abs(*d1R)))
    {
        dtp = min(dtp, 0.01 * (*Rn)/abs(*d1R));
    }
    if (abs(Rnd2R) > epsilon(abs(Rnd2R)))
    {
        dtp = min(dtp, 0.1 * (*Rn)/sqrt(abs(Rnd2R)));
    }
    // Step radius forward
    *Rt = *Rp = (1.0 + dtp/dtm) * (*Rn) - dtp/dtm * Rm + (dtp / (*Rn)) * ( 0.5 * (dtp + dtm) * Rnd2R);
    *d1R = (dtm*dtm * (*Rp) + (-dtm*dtm + dtp*dtp) * (*Rn) - dtp*dtp * Rm)/(dtm * dtp * (dtm + dtp)) + (dtp/(*Rn)) * Rnd2R;
    *dt = dtp;
    // If we are within one timestep of synchronization, move recalculate Rt to match
    if (dtp >= *remain)
    {
        dtp = *remain;
        *Rt = (1.0 + dtp/dtm) * (*Rn) - dtp/dtm * Rm + (dtp /(*Rn)) * ( 0.5 * (dtp + dtm) * Rnd2R);
    }
    // Decrement the timer
    *remain -= dtp;
    return;
}

// Implicit solver for omega_N and alpha_N
__forceinline__ __host__ __device__ double solveOmegaN(doublecomplex * alpha_N, const double PG, const double R, const bub_params_t bub_params)
{
    // Calculate coefficients
    double Rx = R / bub_params.R0;
    double coef1 = bub_params.PG0 /(PG * Rx * Rx * Rx);
    double coef2 = PG * bub_params.rho ;
    double coef3 = 4.0 / (R * R);
    double value1 = -2.0 * bub_params.sig * bub_params.rho / R;
    // Zeroeth step
    doublecomplex Upsilon_N = make_doublecomplex(3.0*bub_params.gam, 0.0) * coef1;
    double mu_eff = bub_params.mu;
    double eta = Upsilon_N.real * coef2 + value1 - mu_eff * mu_eff * coef3;
    double omega_N = sqrt(max(eta, 1.0e-6 * epsilon(eta))) / ( bub_params.rho * R );

    // Implicitly solve alpha_N and omega_N
    for (int i = 0; i < 3; i++)
    {
        *alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
        Upsilon_N = Upsilon(*alpha_N, bub_params.gam) * coef1;
        mu_eff = bub_params.mu + Upsilon_N.imag * PG / (4.0 * (omega_N));
        eta = Upsilon_N.real * coef2 + value1 - mu_eff * mu_eff * coef3;
        omega_N = sqrt(max(eta, 1.0e-6 * epsilon(eta))) / ( bub_params.rho * R );
    }
    *alpha_N = Alpha(R, omega_N, bub_params.coeff_alpha);
    return omega_N;
}

// Calculates and returns alpha_N
static __forceinline__ __host__ __device__ doublecomplex Alpha(const double R, const double omega_N, const double coeff_alpha)
{
    return make_doublecomplex(1.0, 1.0) * sqrt(coeff_alpha / R * omega_N);
}

// Calculates and returns Upsilon_N
static __forceinline__ __host__ __device__ doublecomplex Upsilon(const doublecomplex a, const double gam)
{
    if (abs(a) > 1.0e-1)
    {
        return (3.0 * gam)/(1.0 + (3.0 * ((gam - 1.0) * ((a * coth(a) - 1.0)/(a*a)))));
    }
    else
    {
        return (3.0 * gam)/(1.0 + (3.0 * ((gam - 1.0) * ((1.0/3.0 + a*a * (-1.0/45.0 + a*a * (2.0/945.0 - a*a / 4725.0)))))));
    }
}

// Calculates and returns Lp_N
static __forceinline__ __host__ __device__ doublecomplex solveLp(const doublecomplex a, const double R)
{
    doublecomplex ctmp;
    if (abs(a) > 1.0e-1)
    {
        ctmp = a * coth(a) - 1.0;
        ctmp = (a*a - 3.0 * ctmp)/(a*a*ctmp);
    }
    else
    {
        ctmp = 1.0/5.0 + a*a*(-1.0/175.0+a*a*(2.0/7875.0 - a*a*37.0/3031875.0));
    }
    return R * ctmp;
}

// Calculates and returns PG
__forceinline__ __host__ __device__ double solvePG(const double PGn, const double Rp, const double Rn, const double omega_N, const double dt, const doublecomplex Lp_N, const bub_params_t bub_params)
{
    double R = 0.5 * (Rp + Rn);
    double Rx = R /bub_params.R0;
    double Rpx = Rp/bub_params.R0;
    double Rnx = Rn/bub_params.R0;

    double Lp_NR = Lp_N.real / (Lp_N.real*Lp_N.real + Lp_N.imag*Lp_N.imag);
    double Lp_NI = Lp_N.imag / (Lp_N.real*Lp_N.real + Lp_N.imag*Lp_N.imag);
    double c0 = dt*3.0*(bub_params.gam-1.0)*bub_params.K0 * bub_params.T0 / ( R * bub_params.PG0 );
    double c3 = 1.5 * bub_params.gam * ( Rp - Rn ) / R + c0 * Lp_NR * 0.5 * Rx*Rx*Rx;
    double c1 = 1.0 + c3 - c0 * Lp_NI / (omega_N * dt) * Rpx*Rpx*Rpx;
    double c2 = 1.0 - c3 - c0 * Lp_NI / (omega_N * dt) * Rnx*Rnx*Rnx;
    return ( c2 * PGn + c0 * Lp_NR * bub_params.PG0 ) / c1;
}

/* Bubble Kernels */

// Updates the positional index of an array of bubbles
__global__ void BubbleUpdateIndexKernel()
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    double2 pos;
    // Cache positions to shared memory
    if (index < num_bubbles)
    {
        pos = bubbles_c.pos[index];

        // Convert positions into indices
        bubbles_c.ibm[index] = make_int2(floor(pos.x * grid_c.rdx + 1.0), floor(pos.y * grid_c.rdy + 1.0));
        bubbles_c.ibn[index] = make_int2(floor(pos.x * grid_c.rdx + 0.5), floor(pos.y * grid_c.rdy + 0.5));
    }
}

// Performs interpolation on surrounding mixture cells to determine the pressure acting on a bubble
__global__ void BubbleInterpolationScalarKernel(int p0_width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    int2 ibn;
    double2 pos;
    double Delta_x, Delta_y;
    double PL_p;

    if (index < num_bubbles)
    {
        // Cache to shared memory
        ibn = bubbles_c.ibn[index];
        pos = bubbles_c.pos[index];
        PL_p = 0.0;

        // Run a search around the vicinity of the bubble
        for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++)
        {
            Delta_x = smooth_delta_x(i,pos.x);	// Compute the x component of smooth delta function
            for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++)
            {
                Delta_y = smooth_delta_y(j, pos.y);	// Compute the y component of smooth delta function
                if (	(i >= array_c.ista1m) &&
                        (i <= array_c.iend1m) &&
                        (j >= array_c.jsta1m) &&
                        (j <= array_c.jend1m))  	// Check against grid boundaries
                {
                    PL_p += mixture_c.p0[p0_width * (j - array_c.jsta1m) + (i - array_c.ista1m)] * Delta_x * Delta_y;	// Increment liquid pressure according to influence function (smooth delta)
                }
            }
        }
        bubbles_c.PL_p[index] = PL_p; // Store total liquid pressure
    }
}

// Performs interpolation on surrounding mixture cells to determine the velocity of a bubble
__global__ void BubbleInterpolationVelocityKernel(int vx_width, int vy_width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    int2 ibm, ibn;
    double2 pos;
    double2 v_L = make_double2(0.0, 0.0);
    double Delta_x, Delta_y;

    if (index < num_bubbles)
    {
        // Cache into shared memory
        ibm = bubbles_c.ibm[index];
        ibn = bubbles_c.ibn[index];
        pos = bubbles_c.pos[index];

        // x-component of velocity
        for (int i = ibm.x + bub_params_c.nbs; i <= ibm.x + bub_params_c.nbe; i++)
        {
            Delta_x = smooth_delta_x(i,pos.x);	// compute x-component of smooth delta function
            for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++)
            {
                Delta_y = smooth_delta_y(j,pos.y);	// compute y-component of smooth delta function
                if (	(i >= array_c.ista2n) &&
                        (i <= array_c.iend2n) &&
                        (j >= array_c.jsta2m) &&
                        (j <= array_c.jend2m))  	// check against boundaries
                {
                    v_L.x += mixture_c.vx[vx_width * (j - array_c.jsta2m) + (i - array_c.ista2n)] * Delta_x * Delta_y; // increment velocity
                }
            }
        }
        // y-component of velocity
        for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++)
        {
            Delta_x = smooth_delta_x(i,pos.x);	// compute x-component of smooth delta function
            for (int j = ibm.y + bub_params_c.nbs; j <= ibm.y + bub_params_c.nbe; j++)
            {
                Delta_y = smooth_delta_y(j,pos.y);	// compute y-component of smooth delta function
                if (	(i >= array_c.ista2m) &&
                        (i <= array_c.iend2m) &&
                        (j >= array_c.jsta2n) &&
                        (j <= array_c.jend2n))  	// check against boundaries
                {
                    v_L.y += mixture_c.vy[vy_width * (j - array_c.jsta2n) + (i - array_c.ista2m)] * Delta_x * Delta_y; // increment velocity
                }
            }
        }
        bubbles_c.v_L[index] = v_L; // store the total velocity
    }
}

// 	Calculates the positional movement of a bubble based on it's velocity
__global__ void BubbleMotionKernel()
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    double2 pos;

    //double2 pos0 = make_double2(0.0, 0.0);					// X0 and Y0 boundary
    //double2 pos1 = make_double2(grid_c.X * grid_c.dx, grid_c.Y * grid_c.dy); 	// X1 and Y1 boundary

    if (index < num_bubbles)
    {
        // Move bubbles
        pos = bubbles_c.pos[index] = bubbles_c.pos[index] + bubbles_c.v_B[index] * mix_params_c.dt;

        // Convert positions into indices
        bubbles_c.ibm[index] = make_int2(floor(pos.x * grid_c.rdx + 1.0), floor(pos.y * grid_c.rdy + 1.0));
        bubbles_c.ibn[index] = make_int2(floor(pos.x * grid_c.rdx + 0.5), floor(pos.y * grid_c.rdy + 0.5));
    }
}

// Solves the Rayleigh-Plesset equations for bubble dynamics, to calculate bubble radius
__global__ __launch_bounds__(BUB_RAD_MAX_THREADS, BUB_RAD_MIN_BLOCKS) void BubbleRadiusKernel(int * max_iter)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    double PGn, PL, Rt, omega_N;
    double dTdr_R = 0, SumHeat, SumVis;
    doublecomplex alpha_N, Lp_N;

    double PGp, Rp, Rn, d1Rp, dt_L, remain, time;
    double PC0, PC1, PC2;

    double s0, s1;
    int counter;

    if (index < num_bubbles)
    {
        // Cache bubble parameters
        Rp 	= bubbles_c.R_pn[index];
        Rn 	= bubbles_c.R_nn[index];
        d1Rp 	= bubbles_c.d1_R_n[index];
        PGp	= bubbles_c.PG_n[index];
        dt_L	= bubbles_c.dt_n[index];
        remain	= bubbles_c.re_n[index] + mix_params_c.dt;
        time = -bubbles_c.re_n[index];

        // Calculate coefficients for predicting liquid pressure around the bubble
        PC0 = bubbles_c.PL_n[index] + bub_params_c.PL0;
        PC1 = 0.5*(bubbles_c.PL_p[index]-bubbles_c.PL_m[index])/mix_params_c.dt;
        PC2 = 0.5*(bubbles_c.PL_p[index]+bubbles_c.PL_m[index]-2.0*bubbles_c.PL_n[index])/(mix_params_c.dt * mix_params_c.dt);

        // Reset accumulated variables
        SumHeat = 0.0;
        SumVis = 0.0;
        counter = 0;	// Loop counter

        while (remain > 0.0)
        {
            PGn 	= 	PGp;	// Step the gas pressure forwards
            PL 	= 	PC2 * time * time + PC1 * time + PC0;	// Step the liquid pressure forwards

            solveRayleighPlesset(&Rt, &Rp, &Rn, &d1Rp, PGn, PL, &dt_L, &remain, bub_params_c);	// Solve the reduced order rayleigh-plesset eqn
            time = time + dt_L;	// Increment time step

            omega_N = solveOmegaN (&alpha_N, PGn, Rn, bub_params_c);	// Solve bubble natural frequency

            Lp_N = solveLp(alpha_N, Rn);	// Solve Lp

            PGp = solvePG(PGn, Rp, Rn, omega_N, dt_L, Lp_N, bub_params_c);	// solve for the gas pressure at the next time step

//			PGp = bub_params_c.PG0 * bub_params_c.R03 / (Rp * Rp * Rp);

            // Calculate the the partial derivative dT/dr at the surface of the bubble
            s0 = 0.5*(Rp+Rn);
            s1 = 1 / (Lp_N.real * Lp_N.real + Lp_N.imag * Lp_N.imag) * bub_params_c.T0 / (bub_params_c.PG0 * bub_params_c.R03);
            dTdr_R 	= 	Lp_N.real * s1 * (bub_params_c.PG0 * bub_params_c.R03 - 0.5*(PGp+PGn)*s0*s0*s0) +
                       Lp_N.imag * s1 * (PGp * Rp * Rp * Rp - PGn * Rn * Rn * Rn) / (omega_N * dt_L);

            // Accumulate bubble heat
            if (counter < 0)
            {
                return;
            }

            SumHeat -= 4.0 * Pi * Rp * Rp * bub_params_c.K0 * dTdr_R * dt_L;
            // Accumulate bubble viscous dissipation
            SumVis 	+= 4.0 * Pi * Rp * Rp * 4.0 * bub_params_c.mu * d1Rp / Rp * d1Rp * dt_L;

            counter++;
        }

        // Assign values back to the global memory
        bubbles_c.R_t[index] 	= Rt;
        bubbles_c.R_p[index] 	= Rp;
        bubbles_c.R_n[index] 	= Rn;
        bubbles_c.d1_R_p[index]	= d1Rp;
        bubbles_c.PG_p[index] 	= PGp;
        bubbles_c.dt[index]	= dt_L;
        bubbles_c.re[index]	= remain;
        bubbles_c.Q_B[index]	= (SumHeat + SumVis) / (mix_params_c.dt - remain + bubbles_c.re_n[index]);
        max_iter[index]		= counter;
        //}
    }
    return;
}

/* Mixture Kernels */

// Perform relevant void fraction operations on a cylindrical grid
__global__ void VoidFractionCylinderKernel(int fg_width)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int j2m = j + array_c.jsta2m;

    if ((j2m >= array_c.jsta2m) && (j2m <= array_c.jend2m))
    {
        for (int count = 1 + array_c.ns + array_c.ms; count <= 0; count ++)
        {
            // Mirror void fractions along the center axis
            mixture_c.f_g[fg_width * j + (count - array_c.ista2m)] += mixture_c.f_g[fg_width * j + (1 - count - array_c.ista2m)];
            mixture_c.f_g[fg_width * j + (1 - count - array_c.ista2m)] = mixture_c.f_g[fg_width * j + (count - array_c.ista2m)];
        }
    }
}

// Calculates the void fraction from bubbles
__global__ void VoidFractionReverseLookupKernel(int fg_width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    int2 ibn;
    double2 pos;
    double n_B;
    double R_t;
    double fg_temp;
    double Delta_x, Delta_y;

    if (index < num_bubbles)
    {
        // Cache bubble parameters
        ibn = bubbles_c.ibn[index];
        pos = bubbles_c.pos[index];
        n_B = bubbles_c.n_B[index];
        R_t = bubbles_c.R_t[index];

        // Determine void fraction due to the bubble
        if (plane_wave_c.cylindrical)
        {
            fg_temp =  Pi4r3 * n_B * (R_t * R_t * R_t) * grid_c.rdx * grid_c.rdy / pos.x;
        }
        else
        {
            fg_temp = Pi4r3 * n_B * (R_t * R_t * R_t) * grid_c.rdx * grid_c.rdy;
        }

        // Distribute void fraction using smooth delta function
        for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++)
        {
            Delta_x = smooth_delta_x(i, pos.x);	// Calculate smooth delta function in x
            for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++)
            {
                Delta_y = smooth_delta_y(j, pos.y);	// Calculate smooth delta function in y
                if ((i >= array_c.ista2m) && (i <= array_c.iend2m) && (j >= array_c.jsta2m) && (j <= array_c.jend2m))  	// Check against global boundaries
                {
                    atomicAdd(&mixture_c.f_g[(j - array_c.jsta2m) * fg_width + (i - array_c.ista2m)], fg_temp * Delta_x * Delta_y);	// Use atomics to prevent race conditions
                }
            }
        }
    }
}

// Predicts the void fraction at step n + 1, and stores it in the work array
__global__ void VFPredictionKernel(int fg_width)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int fg_i = fg_width * j + i;

    // Load void fractions for n, n+1, n-1
    if ((i <= array_c.iend2m - array_c.ista2m) && (j <= array_c.jend2m - array_c.jsta2m))
    {
        mixture_c.Work[fg_i] = 3.0 * mixture_c.f_g[fg_i] - 3.0 * mixture_c.f_gn[fg_i] + mixture_c.f_gm[fg_i]; // Calculate future void fraction and store into work array
    }
}

__global__ void VelocityKernel(int vx_width, int vy_width, int rhom_width, int p0_width, int csl_width)
{
    const int tx = threadIdx.x,	ty = threadIdx.y;

    const int i = blockDim.x * blockIdx.x + tx;
    const int j = blockDim.y * blockIdx.y + ty;
    const int in = i + array_c.ista2n;
    const int jn = j + array_c.jsta2n;

    // Compute pitched indices for global memory accesses
    const int rhom_i= rhom_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
    const int p0_i	= p0_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
    const int csl_i = csl_width * (jn - array_c.jsta1m) + (in - array_c.ista1m);
    const int vx_i =  vx_width * (jn - array_c.jsta2m) + (i);
    const int vy_i =  vy_width * (j) + (in - array_c.ista2m);

    // Two sets of boundaries
    const bool ok1 = (in >= array_c.istan) && (in <= array_c.iendn) && (jn >= array_c.jstam) && (jn <= array_c.jendm);	// V_x
    const bool ok2 = (in >= array_c.istam) && (in <= array_c.iendm) && (jn >= array_c.jstan) && (jn <= array_c.jendn);	// V_y

    double s1, s2, s3;

    // Calculate and store X component
    if (ok1)
    {
        s1 = (mixture_c.rho_m[rhom_i] + mixture_c.rho_m[rhom_i + 1]) * 0.5;
        s2 = (-mixture_c.p0[p0_i] + mixture_c.p0[p0_i + 1]) * grid_c.rdx;
        s3 = (mixture_c.c_sl[csl_i] + mixture_c.c_sl[csl_i + 1]) * 0.5;

        s3 = s3 * 0.5 * mix_params_c.dt * sigma_c.nx[i];
        mixture_c.vx[vx_i] = (mixture_c.vx[vx_i] * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);
    }
    // Calculate and store Y component
    if (ok2)
    {
        s1 = (mixture_c.rho_m[rhom_i] + mixture_c.rho_m[rhom_i + rhom_width]) * 0.5;
        s2 = (-mixture_c.p0[p0_i] + mixture_c.p0[p0_i + p0_width]) * grid_c.rdx;
        s3 = (mixture_c.c_sl[csl_i] + mixture_c.c_sl[csl_i + csl_width]) * 0.5;

        s3 = s3 * 0.5 * mix_params_c.dt * sigma_c.ny[j];
        mixture_c.vy[vy_i] = (mixture_c.vy[vy_i] * (1.0 - s3) - mix_params_c.dt / s1 * s2)/(1.0 + s3);
    }
}

// Calculates the velocity at the boundary
__global__ void VelocityBoundaryKernel(int vx_width, int vy_width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    // vx aligned indices
    const int i2n = index + array_c.ista2n;
    const int j2m = index + array_c.jsta2m;
    // vy aligned indices
    const int i2m = index + array_c.ista2m;
    const int j2n = index + array_c.jsta2n;

    // Conditions for vx
    const bool ista2n_to_iend2n = (i2n >= array_c.ista2n) && (i2n <= array_c.iend2n);
    const bool jstam_to_jendm = (j2m >= array_c.jstam) && (j2m <= array_c.jendm);
    // Conditions for vy
    const bool istam_to_iendm = ((i2m >= array_c.istam) && (i2m <= array_c.iendm));
    const bool jsta2n_to_jend2n = ((j2n >= array_c.jsta2n) && (j2n <= array_c.jend2n));

    double s1, s2, xm, ym;
    double vy, vx;

    // VX Boundary
    if (jstam_to_jendm)
    {
        if (PML_c.X0 == 0)
        {
            for (int i = array_c.ms + array_c.ns; i <= -1; i++)
            {
                mixture_c.vx[vx_width * index + i - array_c.ista2n] = mixture_c.vx[vx_width * index - i - array_c.ista2n];
            }
            mixture_c.vx[vx_width * (index) + 0 - array_c.ista2n] = 0.0;
        }
        if (PML_c.X1 == 0)
        {
            for (int i = grid_c.X + 1; i <= grid_c.X + array_c.me + array_c.ne; i++)
            {
                vx = -mixture_c.vx[vx_width * (index) + grid_c.X - (i - grid_c.X) - array_c.ista2n];
                mixture_c.vx[vx_width * (index) + i - array_c.ista2n] = vx;
            }
            mixture_c.vx[vx_width * (index) + grid_c.X - array_c.ista2n] = 0.0;
        }
    }
    if (ista2n_to_iend2n)
    {
        for (int j = 1 + array_c.ms + array_c.ns; j <= 0; j++)
        {
            mixture_c.vx[vx_width * (j - array_c.jsta2m) + index] = mixture_c.vx[vx_width * ( 1 - j - array_c.jsta2m) + index];
        }
        for (int j = grid_c.Y + 1; j <= grid_c.Y + array_c.me + array_c.ne; j++)
        {
            mixture_c.vx[vx_width * (j - array_c.jsta2m) + index] = mixture_c.vx[vx_width * (grid_c.Y - (j - grid_c.Y - 1) - array_c.jsta2m) + index];;
        }
    }
    // VY Boundary
    if (istam_to_iendm)
    {
        if (PML_c.Y0 == 0)
        {
            if ((plane_wave_c.Plane_P == 0) && (plane_wave_c.Focused_P == 0))
            {
                mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index] = 0.0;
            }

            for (int j = array_c.ns + array_c.ms; j <= -1; j++)
            {
                vy = mixture_c.vy[vy_width*(-(j+1) - array_c.jsta2n) + index];
                mixture_c.vy[vy_width*(j - array_c.jsta2n) + index] = vy;
            }

            if (plane_wave_c.Plane_V == 1)
            {
                if ((fmod(tstep_c, (((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq))
                        <=
                        ((double) plane_wave_c.on_wave)/plane_wave_c.freq))
                {
                    mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index]
                    = plane_wave_c.amp * sin(plane_wave_c.omega * tstep_c);
                }
                else
                {
                    mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index] = 0.0;
                }
            }
            if (plane_wave_c.Focused_V == 1)
            {
                ym = 0.0;
                xm = ((double)i2m - 0.5) * grid_c.dx;
                s1 = sqrt(	((plane_wave_c.fp.x - xm)*(plane_wave_c.fp.x - xm)) +
                           ((plane_wave_c.fp.y - ym)*(plane_wave_c.fp.y - ym)));
                if (s1 <= plane_wave_c.f_dist)
                {
                    s1 = max(tstep_c - (plane_wave_c.f_dist - s1) / mix_params_c.cs_inf, 0.0);
                    s2 = (fmod(s1,((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq));
                    if (s2 < ((double)plane_wave_c.on_wave/plane_wave_c.freq))
                    {
                        mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index] = plane_wave_c.amp * sin(plane_wave_c.omega * s2);
                    }
                    else
                    {
                        mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index] = 0.0;
                    }
                }
                else
                {
                    mixture_c.vy[vy_width*(0 - array_c.jsta2n) + index] = 0.0;
                }

            }
        }
        if (PML_c.Y1 == 0)
        {
            mixture_c.vy[vy_width*(grid_c.Y-array_c.jsta2n) + index] = 0.0;
            for (int j = grid_c.Y + 1; j <= grid_c.Y + array_c.ne + array_c.me; j++)
            {
                vy = mixture_c.vy[vy_width*(grid_c.Y - (j - grid_c.Y) - array_c.jsta2n) + index];
                mixture_c.vy[vy_width*(j - array_c.jsta2n) + index] = vy;
            }
        }
    }
    if (jsta2n_to_jend2n)
    {
        for (int i = 1 + array_c.ms + array_c.ns; i <= 0; i++)
        {
            vy = mixture_c.vy[vy_width*index + 1 - i - array_c.ista2m];
            mixture_c.vy[vy_width*index + i - array_c.ista2m] = vy;
        }
        for (int i = grid_c.X + 1; i <= grid_c.X + array_c.me + array_c.ne; i++)
        {
            vy = mixture_c.vy[vy_width*i + grid_c.X - (i - grid_c.X - 1) - array_c.ista2m];
            mixture_c.vy[vy_width*index + i - array_c.ista2m] = vy;
        }
    }

}

// Calculates the mixture pressure
__global__ void MixturePressureKernel(int vx_width, int vy_width, int fg_width, int rhol_width, int csl_width, int p0_width, int p_width, int Work_width)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int i1m = i + array_c.ista1m;
    const int j1m = j + array_c.jsta1m;

    const int fg_i = fg_width * (j1m - array_c.jsta2m) + i1m - array_c.ista2m;
    const int rhol_i = rhol_width * j + i;
    const int csl_i = csl_width * j + i;
    const int p_i = p_width * j + i;
    const int vx_i = vx_width * (j1m - array_c.jsta2m) + i1m - array_c.ista2n;
    const int vy_i = vy_width * (j1m - array_c.jsta2n) + i1m - array_c.ista2m;
    const int xu_i = i1m - array_c.ista2n;
    const int Work_i = Work_width * j + i;

    const bool ok = ((i1m >= array_c.istam) && (i1m <= array_c.iendm) && (j1m >= array_c.jstam) && (j1m <= array_c.jendm));

    double2 s1, s2;
    double s3, s4, s5;
    double2 p;

    if (ok)
    {
        // Determine the velocity gradient in x (depends on whether the cylindrical condition is active or not)
        if (plane_wave_c.cylindrical)
        {
            s1.x = (-gridgen_c.xu[xu_i - 1] * mixture_c.vx[vx_i - 1] + gridgen_c.xu[xu_i] * mixture_c.vx[vx_i]) * grid_c.rdx * gridgen_c.rxp[i1m - array_c.ista2m];
        }
        else
        {
            s1.x = (-mixture_c.vx[vx_i - 1] + mixture_c.vx[vx_i]) * grid_c.rdx;
        }
        // Determine the velocity gradient in y
        s1.y = (-mixture_c.vy[vy_i - vy_width] + mixture_c.vy[vy_i]) * grid_c.rdy;
        // Calculate coefficients
        s3 = mix_params_c.dt * mixture_c.rho_l[rhol_i] * mixture_c.c_sl[csl_i] * mixture_c.c_sl[csl_i];
        s4 = 0.5 * mix_params_c.dt * mixture_c.c_sl[csl_i];
        s5 = -(mixture_c.f_g[fg_i] - mixture_c.f_gn[fg_i])/ mix_params_c.dt / (1.0-0.5*(mixture_c.f_g[fg_i] + mixture_c.f_gn[fg_i]));
        // Calculate PML components
        s2 = make_double2(s4 * sigma_c.mx[i], s4 * sigma_c.my[j]);

        // Vectorized version of P calculation
        mixture_c.p[p_i] = p = (mixture_c.pn[p_i] * (1.0 - s2) - s3 * (s1 + 0.5 * s5) / (1.0 + s2));

        // Take the difference of the new pressure and the old pressure, store it into the temporary array
        mixture_c.Work[Work_i] = abs(mixture_c.p0[p_i] - (p.x + p.y));
        // Store the new pressure
        mixture_c.p0[p_i] = p.x + p.y;
    }

}

// Calculates the mixture pressure at the boundaries
__global__ void MixtureBoundaryPressureKernel(int p0_width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    const int i1m = index + array_c.ista1m;
    const int j1m = index + array_c.jsta1m;

    const double pamp = mix_params_c.rho_inf * mix_params_c.cs_inf * plane_wave_c.amp;

    double s1, s2;

    if ((i1m >= array_c.istam) && (i1m <= array_c.iendm))
    {
        if (PML_c.Y0 == 0)
        {
            for (int j = array_c.ms; j <= 0; j++)
            {
                mixture_c.p0[p0_width * (j - array_c.jsta1m) + index] = mixture_c.p0[p0_width * (1 - j - array_c.jsta1m) + index];
            }
            if (plane_wave_c.Plane_P == 1)
            {

                if (fmod(tstep_c, ((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq)
                        <=
                        ((double)plane_wave_c.on_wave)/plane_wave_c.freq)
                {
                    mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = pamp * sin(plane_wave_c.omega * tstep_c);
                }
                else
                {
                    mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
                }
            }
            if (plane_wave_c.Focused_P == 1)
            {
                double ym = -0.5 * grid_c.dy;
                double xm = (i1m - 0.5) * grid_c.dx;
                s1 = sqrt((plane_wave_c.fp.x - xm) * (plane_wave_c.fp.x - xm) + (plane_wave_c.fp.y - ym) * (plane_wave_c.fp.y - ym));
                if (s1 <= plane_wave_c.f_dist)
                {
                    s1 = max(tstep_c - (plane_wave_c.f_dist - s1)/mix_params_c.cs_inf, 0.0);
                    s2 = fmod(s1, ((double)plane_wave_c.on_wave + (double)plane_wave_c.off_wave)/plane_wave_c.freq);
                    if (s2 < (double)plane_wave_c.on_wave/plane_wave_c.freq)
                    {
                        mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = pamp * sin(plane_wave_c.omega * s2);
                    }
                    else
                    {
                        mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
                    }
                }
                else
                {
                    mixture_c.p0[p0_width * (0 - array_c.jsta1m) + index] = 0.0;
                }
            }
        }
        if (PML_c.Y1 == 0)
        {
            for (int j = grid_c.X; j <= grid_c.X + array_c.me; j++)
            {
                mixture_c.p0[p0_width * (j - array_c.jsta1m) + index] = mixture_c.p0[p0_width * (grid_c.Y - (j - grid_c.Y - 1 - array_c.jsta1m)) + index];
            }
        }
    }
    if ((j1m >= array_c.jsta1m) && (j1m <= array_c.jend1m))
    {
        if (PML_c.X0 == 0)
        {
            for (int i = array_c.ms; i <= 0; i++)
            {
                mixture_c.p0[p0_width * index + i - array_c.ista1m] = mixture_c.p0[p0_width * index + 1 - i - array_c.ista1m];
            }
        }
        if (PML_c.X1 == 0)
        {
            for (int i = grid_c.X + 1; i <= grid_c.X + array_c.me; i++)
            {
                mixture_c.p0[p0_width * index + i - array_c.ista1m] = mixture_c.p0[p0_width * index + (grid_c.X - (i - grid_c.X - 1) - array_c.ista1m)];
            }
        }
    }
}

// Calculates the conductivity of the mixture based on void fraction
__global__ void MixtureKMKernel(int km_width, int T_width, int fg_width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int i2m = i + array_c.ista2m;
    const int j2m = j + array_c.jsta2m;

    const bool ok = ((i2m <= array_c.iend2m) && (j2m <= array_c.jend2m));

    if (ok)
    {
        mixture_c.k_m[j * km_width + i] = thermal_conductivity_water(mixture_c.T[j * T_width + i] + mix_params_c.T_inf) * (1.0 - 3.0 * mixture_c.f_g[j * fg_width + i] / (2.0 + mixture_c.f_g[j * fg_width + i]));
    }
}

// Calculates heat transfer from bubbles to mixture
__global__ void BubbleHeatKernel(int Work_width)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_bubbles)
    {
        double	Q_temp;
        int2	ibn = bubbles_c.ibn[index];
        double2	pos = bubbles_c.pos[index];
        double	n_B = bubbles_c.n_B[index];
        double	Q_B = bubbles_c.Q_B[index];

        if (plane_wave_c.cylindrical)
        {
            Q_temp = n_B * Q_B * grid_c.rdx * grid_c.rdy / pos.x;
        }
        else
        {
            Q_temp = n_B * Q_B * grid_c.rdx * grid_c.rdy;
        }

        double Delta_x, Delta_y;
        for (int i = ibn.x + bub_params_c.mbs; i <= ibn.x + bub_params_c.mbe; i++)
        {
            Delta_x = smooth_delta_x(i, pos.x);
            for (int j = ibn.y + bub_params_c.mbs; j <= ibn.y + bub_params_c.mbe; j++)
            {
                Delta_y = smooth_delta_y(j, pos.y);
                if ((i >= array_c.ista2m) && (i <= array_c.iend2m) && (j >= array_c.jsta2m) && (j <= array_c.jend2m))
                {
                    atomicAdd(&mixture_c.Work[(j - array_c.jsta2m) * Work_width + (i - array_c.ista2m)], Q_temp * Delta_x * Delta_y);
                    //atomicAdd(&mixture_c.Work[(j - array_c.jsta2m) * Work_width + (i - array_c.ista2m)], 1);
                }
            }
        }
    }
}

// Calculates the mixture energy
__global__ void MixtureEnergyKernel(int km_width, int T_width, int Ex_width, int Ey_width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int i2m = i + array_c.ista2m;
    const int j2m = j + array_c.jsta2m;

    __syncthreads();

    if ((i2m >= array_c.ista1n) && (i2m <= array_c.iend1n) && (j2m >= array_c.jsta1m) && (j2m <= array_c.jend1m))
    {
        double s0 = 2.0 * mixture_c.k_m[j * km_width + i] * mixture_c.k_m[j * km_width + i + 1] / (mixture_c.k_m[j * km_width + i] + mixture_c.k_m[j * km_width + i + 1]);
        double s1 = ( -mixture_c.T[j * T_width + i] + mixture_c.T[j * T_width + i + 1]) * grid_c.rdx;

        mixture_c.Ex[(j2m - array_c.jsta1m) * Ex_width + i2m - array_c.ista1n] = s0 * s1;
    }

    if ((i2m >= array_c.ista1m) && (i2m <= array_c.iend1m) && (j2m >= array_c.jsta1n) && (j2m <= array_c.jend1n))
    {
        double s0 = 2.0 * mixture_c.k_m[j * km_width + i] * mixture_c.k_m[(j+1) * km_width + i] / (mixture_c.k_m[j * km_width + i] + mixture_c.k_m[(j+1) * km_width + i]);
        double s1 = ( -mixture_c.T[j * T_width + i] + mixture_c.T[(j+1) * T_width + i]) * grid_c.rdy;

        mixture_c.Ey[(j2m - array_c.jsta1n) * Ey_width + i2m - array_c.ista1m] = s0 * s1;
    }
}

// Calculates the mixture temperature
__global__ void MixtureTemperatureKernel(int T_width, int Ex_width, int Ey_width, int p_width, int rhom_width, int Cpm_width, int Work_width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int im = i + array_c.ista2m;
    const int jm = j + array_c.jsta2m;

    const int p_i = (jm - array_c.jsta1m) * p_width + im - array_c.ista1m;
    const int Ex_i = (jm - array_c.jsta1m) * Ex_width + im - array_c.ista1n;
    const int Ey_i = (jm - array_c.jsta1n) * Ey_width + im - array_c.ista1m;

    const bool ok = ((im >= array_c.istam) && (im <= array_c.iendm) && (jm >= array_c.jstam) && (jm <= array_c.jendm));

    if (ok)
    {
        double s0 = (mixture_c.p[p_i].x + mixture_c.p[p_i].y - mixture_c.pn[p_i].x - mixture_c.pn[p_i].y) / mix_params_c.dt;
        s0 = 2.0 * (0.22) * 0.115129255 / ( mix_params_c.rho_inf * mix_params_c.cs_inf * plane_wave_c.omega * plane_wave_c.omega) * s0*s0;
        double Q = mixture_c.Work[(jm - array_c.jsta2m) * Work_width + im - array_c.ista2m] + s0;

        double s1 = (mixture_c.Ex[Ex_i] - mixture_c.Ex[Ex_i - 1]) * grid_c.rdx;
        double s2 = (mixture_c.Ey[Ey_i] - mixture_c.Ey[Ey_i - Ey_width]) * grid_c.rdy;
        mixture_c.T[(jm - array_c.jsta2m) * T_width + im - array_c.ista2m] += mix_params_c.dt * (s1 + s2 + Q ) / (mixture_c.rho_m[(jm - array_c.jsta1m) * rhom_width + im - array_c.ista1m] * mixture_c.C_pm[(jm - array_c.jsta1m) * Cpm_width + im - array_c.ista1m]);
    }
}

// Calculates the temperature at the boundary of the mixture
__global__ void MixtureBoundaryTemperatureKernel(int T_width)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int i2m = index + array_c.ista2m;
    const int j2m = index + array_c.jsta2m;

    if ((PML_c.X0 == 0) && (j2m >= array_c.jstam) && (j2m <= array_c.jendm))
    {
        for (int i = 1 + array_c.ns + array_c.ms; i <= 0; i++)
        {
            mixture_c.T[index * T_width + i - array_c.ista2m] = mixture_c.T[index * T_width + 1 - i - array_c.ista2m];
        }
    }
    __syncthreads();
    if ((PML_c.X1 == 0) && (j2m >= array_c.jsta1m) && (j2m <= array_c.jend1m))
    {
        for (int i = grid_c.X + 1; i <= grid_c.X + array_c.ne + array_c.me; i++)
        {
            mixture_c.T[index * T_width + i - array_c.ista2m] = mixture_c.T[index * T_width + grid_c.X - (i - grid_c.X - 1) - array_c.ista2m];
        }
    }
    __syncthreads();
    if ((PML_c.Y0 == 0) && (i2m <= array_c.iend2m))
    {
        for (int j = 1 + array_c.ns + array_c.ms; j <= 0; j++)
        {
            mixture_c.T[(j - array_c.jsta2m) * T_width + index] = mixture_c.T[(1 - j - array_c.jsta2m) * T_width + index];
        }
    }
    __syncthreads();
    if ((PML_c.Y1 == 0) && (i2m <= array_c.iend2m))
    {
        for (int j = grid_c.Y; j <= grid_c.Y + array_c.ne + array_c.me; j++)
        {
            mixture_c.T[(j - array_c.jsta2m) * T_width + index] = mixture_c.T[(grid_c.Y - (j - grid_c.Y - 1) - array_c.jsta2m) * T_width + index];
        }
    }
}


// Calculates various mixture properties:
//	Density (liquid and mixture)
//	Speed of sound
//	Specific heat capacity

__global__ void MixturePropertiesKernel(int rhol_width, int rhom_width, int csl_width, int cpm_width, int fg_width, int T_width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int i1m = i + array_c.ista1m;
    const int j1m = j + array_c.jsta1m;

    if ((i1m <= array_c.iend1m) && (j1m <= array_c.jend1m))
    {
        mixture_c.rho_l[j * rhol_width + i] = mix_params_c.rho_inf;
        mixture_c.c_sl[j * csl_width + i] = mix_params_c.cs_inf;
        mixture_c.rho_m[j * rhom_width + i] = (1.0 - mixture_c.f_g[(j1m - array_c.jsta2m) * fg_width + i1m - array_c.ista2m]) * mix_params_c.rho_inf;
        mixture_c.C_pm[j * cpm_width + i] = specific_heat_water(mixture_c.T[(j1m - array_c.jsta2m) * T_width + i1m - array_c.ista2m] + mix_params_c.T_inf);
    }
}






///////////////////////////////////////////////////////////////////////////////////////////////////
//	Wrapper Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

int update_bubble_indices(cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    BubbleUpdateIndexKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> ();

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Failed to update bubble index");
    return 0;
}

int calculate_void_fraction(mixture_t mixture_htod, plane_wave_t *plane_wave, int f_g_width, int f_g_pitch, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimVFCBlock(LINEAR_BLOCK_SIZE);
    dim3 dimVFCGrid((max(i2m, j2m) + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE);

    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    CUDA_SAFE_CALL(cudaMemset2D(mixture_htod.f_g, f_g_pitch, 0, i2m * sizeof(double), j2m));
    cudaThreadSynchronize();

    VoidFractionReverseLookupKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> (f_g_width);
    if (plane_wave->cylindrical)
    {
        VoidFractionCylinderKernel <<< dimVFCGrid, dimVFCBlock, 0, stream[0] >>> (f_g_width);
    }

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Failed to calculate void fraction");
    return 0;
}

int synchronize_void_fraction(mixture_t mixture_htod, size_t f_g_pitch, cudaStream_t stream[], cudaEvent_t stop[])
{

    CUDA_SAFE_CALL(cudaMemcpy2D(mixture_htod.f_gn, f_g_pitch, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(mixture_htod.f_gm, f_g_pitch, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice));

    return 0;
}

int store_variables(mixture_t mixture_htod, bubble_t bubbles_htod, int f_g_width, int p_width, int Work_width, size_t f_g_pitch, size_t Work_pitch, size_t p_pitch, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dim2mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
    dim3 dim2mGrid((i2m + TILE_BLOCK_WIDTH - 1) / TILE_BLOCK_WIDTH,
                   (j2m + TILE_BLOCK_HEIGHT - 1) / TILE_BLOCK_HEIGHT);

    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    CUDA_SAFE_CALL(cudaMemset2D(mixture_htod.Work, Work_pitch, 0, i2m * sizeof(double), j2m));
    cudaThreadSynchronize();

    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.R_pn, bubbles_htod.R_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.R_nn, bubbles_htod.R_n, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.d1_R_n, bubbles_htod.d1_R_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.PG_n, bubbles_htod.PG_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.PL_m, bubbles_htod.PL_n, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.PL_n, bubbles_htod.PL_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.dt_n, bubbles_htod.dt, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.re_n, bubbles_htod.re, sizeof(double)*numBubbles, cudaMemcpyDeviceToDevice));

    // Void fraction (fg) prediction, store it in the work array
    VFPredictionKernel <<< dim2mGrid, dim2mBlock, 0, stream[0] >>> (f_g_width);
    cudaThreadSynchronize();
    // Store variables
    CUDA_SAFE_CALL(cudaMemcpy2D(	mixture_htod.pn, p_pitch, mixture_htod.p, p_pitch, sizeof(double2)*i1m, j1m, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(	mixture_htod.f_gm, f_g_pitch, mixture_htod.f_gn, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy2D(	mixture_htod.f_gn, f_g_pitch, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice));

    // Store the predicted void fractions (fg)
    CUDA_SAFE_CALL(cudaMemcpy2D(	mixture_htod.f_g, f_g_pitch, mixture_htod.Work, Work_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToDevice));

    return 0;
}

int calculate_velocity_field(int vx_width, int vy_width, int rho_m_width, int p0_width, int c_sl_width, cudaStream_t stream[], cudaEvent_t stop[])
{

    dim3 dimVelocityBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
    dim3 dimVelocityGrid((max(i2m, i2n) + TILE_BLOCK_WIDTH - 1) / (TILE_BLOCK_WIDTH),
                         (max(j2m, j2n) + TILE_BLOCK_HEIGHT - 1) / (TILE_BLOCK_HEIGHT));

    dim3 dimVBBlock(LINEAR_BLOCK_SIZE);
    dim3 dimVBGrid((max(max(i2m,i2n),max(j2m,j2n)) + LINEAR_BLOCK_SIZE - 1)/ LINEAR_BLOCK_SIZE);

    VelocityKernel <<< dimVelocityGrid, dimVelocityBlock, 0, stream[0] >>> (vx_width, vy_width, rho_m_width, p0_width, c_sl_width);
    VelocityBoundaryKernel <<< dimVBGrid, dimVBBlock, 0, stream[0] >>> (vx_width, vy_width);

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Failed to calculate velocity");
    return 0;
}

int bubble_motion(bubble_t bubbles_htod, int vx_width, int vy_width, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    // Calculate each bubble's interpolated liquid velocity
    BubbleInterpolationVelocityKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> (vx_width, vy_width);
    // Calculate (read: copy) the bubble velocity from liquid velocity
    CUDA_SAFE_CALL(cudaMemcpy(bubbles_htod.v_B, bubbles_htod.v_L, sizeof(double2)*numBubbles, cudaMemcpyDeviceToDevice));
    // Move the bubbles and update the bubble midpoint/nodepoint indices
    BubbleMotionKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> ();

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Failed to move bubbles");
    return 0;
}

double calculate_pressure_field(mixture_t mixture_h, mixture_t mixture_htod, double P_inf, int p0_width, int p_width, int f_g_width, int vx_width, int vy_width, int rho_l_width, int c_sl_width, int Work_width, size_t Work_pitch, cudaStream_t stream[], cudaEvent_t stop[])
{
    double resimax = 0.0;

    dim3 dimMBPBlock(LINEAR_BLOCK_SIZE);
    dim3 dimMBPGrid((max(i1m, j1m) + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    dim3 dim1mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
    dim3 dim1mGrid((i1m + TILE_BLOCK_WIDTH - 1)/TILE_BLOCK_WIDTH,
                   (j1m + TILE_BLOCK_HEIGHT - 1)/TILE_BLOCK_HEIGHT);

    CUDA_SAFE_CALL(cudaMemset2D(mixture_htod.Work, Work_pitch, 0, i2m * sizeof(double), j2m));

    MixturePressureKernel <<< dim1mGrid, dim1mBlock, 0, stream[0] >>> (vx_width, vy_width, f_g_width, rho_l_width, c_sl_width, p0_width, p_width, Work_width);
    cudaStreamSynchronize(stream[0]);

    MixtureBoundaryPressureKernel <<< dimMBPGrid, dimMBPBlock, 0, stream[1] >>> (p0_width);

    CUDA_SAFE_CALL(cudaMemcpy2D(mixture_h.Work, sizeof(double)*i2m, mixture_htod.Work, Work_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToHost));

    cudaStreamSynchronize(stream[0]);
    resimax = 0;
    for (int i = 0; i < i2m * j2m; i++)
    {
        if (resimax < mixture_h.Work[i])
        {
            resimax = mixture_h.Work[i];
        }
    }
    resimax /= P_inf;

    cudaStreamSynchronize(stream[1]);
    checkCUDAError("Failed to calculate pressure");
    return resimax;
}

int interpolate_bubble_pressure(int p0_width, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    BubbleInterpolationScalarKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> (p0_width);
    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Failed to calculate PL for bubbles");

    return 0;
}

int calculate_temperature(mixture_t mixture_htod, bub_params_t *bub_params, int k_m_width, int T_width, int f_g_width, int Ex_width, int Ey_width, int p_width, int rho_m_width, int C_pm_width, int Work_width, size_t Work_pitch, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimBubbleBlock(LINEAR_BLOCK_SIZE);
    dim3 dimBubbleGrid((numBubbles + LINEAR_BLOCK_SIZE - 1) / (LINEAR_BLOCK_SIZE));

    dim3 dim2mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
    dim3 dim2mGrid((i2m + TILE_BLOCK_WIDTH - 1) / TILE_BLOCK_WIDTH,
                   (j2m + TILE_BLOCK_HEIGHT - 1) / TILE_BLOCK_HEIGHT);

    dim3 dimMBTBlock(LINEAR_BLOCK_SIZE);
    dim3 dimMBTGrid((max(i2m, j2m) + LINEAR_BLOCK_SIZE - 1)/LINEAR_BLOCK_SIZE);

    CUDA_SAFE_CALL(cudaMemset2D(mixture_htod.Work, Work_pitch, 0, i2m * sizeof(double), j2m));
    cudaThreadSynchronize();

    if (bub_params->enabled)
    {
        BubbleHeatKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[1] >>> (Work_width);
    }
    MixtureKMKernel <<< dim2mGrid, dim2mBlock, 0, stream[0] >>> (k_m_width, T_width, f_g_width);
    MixtureEnergyKernel <<< dim2mGrid, dim2mBlock, 0, stream[0] >>> (k_m_width, T_width, Ex_width, Ey_width);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    MixtureTemperatureKernel <<< dim2mGrid, dim2mBlock, 0, stream[0] >>> (T_width, Ex_width, Ey_width, p_width, rho_m_width, C_pm_width, Work_width);

    MixtureBoundaryTemperatureKernel <<< dimMBTGrid, dimMBTBlock, 0, stream[0] >>> (T_width);

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Mixture boundary temperature");

    return 0;
}

int calculate_properties(int rho_l_width, int rho_m_width, int c_sl_width, int C_pm_width, int f_g_width, int T_width, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dim1mBlock(TILE_BLOCK_WIDTH, TILE_BLOCK_HEIGHT);
    dim3 dim1mGrid((i1m + TILE_BLOCK_WIDTH - 1)/TILE_BLOCK_WIDTH,
                   (j1m + TILE_BLOCK_HEIGHT - 1)/TILE_BLOCK_HEIGHT);

    MixturePropertiesKernel <<< dim1mGrid, dim1mBlock, 0, stream[0] >>> (rho_l_width, rho_m_width, c_sl_width, C_pm_width, f_g_width, T_width);

    cudaStreamSynchronize(stream[0]);
    checkCUDAError("Mixture properties");

    return 0;
}

int solve_bubble_radii(bubble_t bubbles_htod, cudaStream_t stream[], cudaEvent_t stop[])
{
    dim3 dimBubbleBlock(BUB_RAD_MAX_THREADS);
    dim3 dimBubbleGrid((numBubbles + BUB_RAD_MAX_THREADS - 1) / BUB_RAD_MAX_THREADS);

    thrust::device_vector<int> max_iter_d(numBubbles);

    BubbleRadiusKernel <<< dimBubbleGrid, dimBubbleBlock, 0, stream[0] >>> (thrust::raw_pointer_cast(&max_iter_d[0]));

    cudaStreamSynchronize(stream[0]);
#ifdef _DEBUG_
    checkCUDAError("Bubble Radius");
#endif

    return thrust::reduce(max_iter_d.begin(), max_iter_d.end(), (int) 0, thrust::maximum<int>());
}
