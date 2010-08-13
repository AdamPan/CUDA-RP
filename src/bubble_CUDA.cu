// CUDA solver for Rayleigh Plesset


#include "bubble_CUDA.h"
#include "bubble_CUDA_kernel.cuh"

using namespace thrust;


// Mathematical constants
const double 	Pi_h =  acos(-1.0);
const double	Pi4r3_h = Pi_h * 4.0/3.0;

mixture_t	mixture_h, mixture_htod;
bubble_t	bubbles_h, bubbles_htod;

grid_gen	grid_h, grid_htod;
sigma_t		sigma_h, sigma_htod;

size_t T_pitch, P_pitch, p0_pitch, p_pitch, pn_pitch, vx_pitch, vy_pitch, c_sl_pitch, rho_m_pitch, rho_l_pitch, f_g_pitch, f_gn_pitch, f_gm_pitch, k_m_pitch, C_pm_pitch, Work_pitch, Ex_pitch, Ey_pitch;

int T_width, P_width, p0_width, p_width, pn_width, vx_width, vy_width, c_sl_width, rho_m_width, rho_l_width, f_g_width, f_gn_width, f_gm_width, k_m_width, C_pm_width, Work_width, Ex_width, Ey_width;

int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;
int numBubbles = 0;



/**************************************************************
 *                      Host Functions                        *
 **************************************************************/

host_vector<solution_space> solve_bubbles(	grid_t	*grid_size,
						PML_t	*PML,
						sim_params_t	*sim_params,
						bub_params_t	*bub_params,
						plane_wave_t	*plane_wave,
						debug_t		*debug,
						int argc,
						char ** argv){
	// Clear terminal for output
	if(system("clear")){exit(EXIT_FAILURE);}


	// Variable Declaration
	// Storage variable for full solution (mixture and bubble data)
	host_vector<solution_space> solution;

	// Mixture Parameters
	mix_params_t	*mix_params = (mix_params_t*) calloc(1, sizeof(mix_params_t));

	// Array Indices
	array_index_t	*array_index = (array_index_t *) calloc(1, sizeof(array_index_t));

	// Variables needed for control structures
	unsigned int nstep = 0, save_count = 0;
	double tstep = 0.0, tstepx = 0.0;
	int loop;
	double resimax;
	double s1, s2;

	int max_iter;

	CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleYield));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(BubbleUpdateIndexKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(BubbleInterpolationScalarKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(BubbleInterpolationVelocityKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(BubbleRadiusKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(VoidFractionCylinderKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(VoidFractionReverseLookupKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(VFPredictionKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(VelocityKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(VelocityBoundaryKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixturePressureKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixtureBoundaryPressureKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixtureKMKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(BubbleHeatKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixtureEnergyKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixtureTemperatureKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixtureBoundaryTemperatureKernel, cudaFuncCachePreferL1));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(MixturePropertiesKernel, cudaFuncCachePreferL1));

	// Initialize Variables
	printf("Initializing\n");
	if(initialize_variables(grid_size, PML, sim_params, plane_wave, array_index, mix_params, bub_params)){
		exit(EXIT_FAILURE);
	}

	// Allocate memory on the device
	printf("Allocating Memory\n\n");
	if(initialize_CUDA_variables(grid_size, PML, sim_params, plane_wave,array_index, mix_params, bub_params)){
		exit(EXIT_FAILURE);
	}


	// Initialization kernels

	printf("Running Ininitalization Kernels\n\n");

	if(bub_params->enabled){
		if(update_bubble_indices()){
		exit(EXIT_FAILURE);
		}
	}

	if(bub_params->enabled){
		if(calculate_void_fraction(mixture_htod, plane_wave, f_g_width, f_g_pitch)){exit(EXIT_FAILURE);}
		if(synchronize_void_fraction(mixture_htod, f_g_pitch)){exit(EXIT_FAILURE);}
	}

	// Zero all counters

	nstep = 0;
	tstep = 0.0;
	tstepx = 0.0;
	save_count = 0;

	// Stop for the user to check params

	printf("Simulation ready! Press any key to continue\n");
	if(system("clear")){exit(EXIT_FAILURE);}

	/************************
	 * Main simulation loop	*
	 ************************/
	printf("Running the simulation\n\n");
	do{
		nstep++;

		// Accurate time addition
		s1	=	tstep;
		tstep	=	tstep + (*mix_params).dt + tstepx;
		s2	=	tstep - s1;
		tstepx	=	(*mix_params).dt + tstepx - s2;

		cudaMemcpyToSymbol(tstep_c, &tstep, sizeof(double));
		cudaThreadSynchronize();
		checkCUDAError("Copy tstep");

		// Store Bubble and Mixture, and predict void fraction
		if(store_variables(mixture_htod, bubbles_htod, f_g_width, p_width, Work_width, f_g_pitch, Work_pitch, p_pitch)){exit(EXIT_FAILURE);}

		// Update mixture velocity
		if(calculate_velocity_field(vx_width, vy_width, rho_m_width, p0_width, c_sl_width)){exit(EXIT_FAILURE);}

		if(bub_params->enabled){
			// Move the bubbles
			if(bubble_motion(bubbles_htod, vx_width, vy_width)){exit(EXIT_FAILURE);}
		}

		resimax = calculate_pressure_field(mixture_h, mixture_htod, mix_params->P_inf, p0_width, p_width, f_g_width, vx_width, vy_width, rho_l_width, c_sl_width, Work_width, Work_pitch);

		loop = 0;

		if(bub_params->enabled){
			while (resimax > 1.0e-7f){
				loop++;

				// Find bubble pressure
				if(interpolate_bubble_pressure(p0_width)){exit(EXIT_FAILURE);}

				// Solve Rayleigh-Plesset Equations
				//max_iter = solve_bubble_radii_host(bubbles_h, bubbles_htod, *bub_params, *mix_params);
				max_iter = solve_bubble_radii(bubbles_htod);

				// Calculate Void Fraction
				if(calculate_void_fraction(mixture_htod, plane_wave, f_g_width, f_g_pitch)){exit(EXIT_FAILURE);}

				// Calculate Pressure
				resimax = calculate_pressure_field(mixture_h, mixture_htod, mix_params->P_inf, p0_width, p_width, f_g_width, vx_width, vy_width, rho_l_width, c_sl_width, Work_width, Work_pitch);

				#ifdef _DEBUG_
				printf("\033[A\033[2K");
				printf("Simulation step %5i subloop %5i \t resimax = %4.2E, inner loop executed %i times.\n", nstep, loop, resimax, max_iter);
				#endif
				if (loop == 10000) {printf("Premature Termination at nstep %i, subloop %i\n\n", nstep, loop); break;}
			}
		}

		// Calculate mixture temperature
		if(calculate_temperature(bub_params, k_m_width, T_width, f_g_width, Ex_width, Ey_width, p_width, rho_m_width, C_pm_width, Work_width)){exit(EXIT_FAILURE);}
		// Calculate mixture properties
		if(calculate_properties(rho_l_width, rho_m_width, c_sl_width, C_pm_width, f_g_width, T_width)){exit(EXIT_FAILURE);}

		// Save data at intervals
		if((((int)nstep) % ((int)sim_params->DATA_SAVE) == 0)){
			solution.resize(save_count+1);
			if (debug->p0) solution[save_count].p0 = (double*)calloc(m1Vol, sizeof(double));
			if (debug->fg) solution[save_count].f_g = (double*)calloc(m2Vol, sizeof(double));
			if (debug->T)solution[save_count].T = (double*)calloc(m2Vol, sizeof(double));
			if (debug->vxy) solution[save_count].vx = (double*)calloc(v_xVol, sizeof(double));
			if (debug->vxy) solution[save_count].vy = (double*)calloc(v_yVol, sizeof(double));
			if (debug->pxy)solution[save_count].p = (double2*)calloc(m1Vol, sizeof(double2));
			if (debug->bubbles)solution[save_count].pos = (double2*)calloc(numBubbles, sizeof(double2));
			if (debug->bubbles)solution[save_count].R_t = (double*)calloc(numBubbles, sizeof(double));
			if (debug->bubbles)solution[save_count].PG_p = (double*)calloc(numBubbles, sizeof(double));
			if (debug->p0)cudaMemcpy2D(	solution[save_count].p0, sizeof(double)*i1m, mixture_htod.p0, p0_pitch, sizeof(double)*i1m, j1m, cudaMemcpyDeviceToHost);
			if (debug->fg)cudaMemcpy2D(	solution[save_count].f_g, sizeof(double)*i2m, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToHost);
			if (debug->T)cudaMemcpy2D(	solution[save_count].T, sizeof(double)*i2m, mixture_htod.T, T_pitch, sizeof(double)*i2m, j2m, 	cudaMemcpyDeviceToHost);
			if (debug->vxy)cudaMemcpy2D(	solution[save_count].vx, sizeof(double)*i2n, mixture_htod.vx, vx_pitch, sizeof(double)*i2n, j2m, cudaMemcpyDeviceToHost);
			if (debug->vxy)cudaMemcpy2D(	solution[save_count].vy, sizeof(double)*i2m, mixture_htod.vy, vy_pitch, sizeof(double)*i2m, j2n, cudaMemcpyDeviceToHost);
			if (debug->pxy)cudaMemcpy2D(	solution[save_count].p, sizeof(double2)*i1m, mixture_htod.p, p_pitch, sizeof(double2)*i1m, j1n, cudaMemcpyDeviceToHost);
			if (debug->bubbles)cudaMemcpy(	solution[save_count].pos, bubbles_htod.pos, sizeof(double2)*numBubbles, cudaMemcpyDeviceToHost);
			if (debug->bubbles)cudaMemcpy(	solution[save_count].R_t, bubbles_htod.R_t, sizeof(double)*numBubbles, cudaMemcpyDeviceToHost);
			if (debug->bubbles)cudaMemcpy(	solution[save_count].PG_p, bubbles_htod.PG_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToHost);
		#ifdef _DEBUG_
			if(system("clear")){exit(EXIT_FAILURE);}
		#endif
			printf("Simulation step : %i\ttstep : %4.2E\tdt : %4.2E\n",
					nstep,
					tstep,
					mix_params->dt);
		#ifdef _DEBUG_
			double a = debug->display - 1;
			if(debug->bubbles){
				for (double i = 0; i <= numBubbles - 1; i += (numBubbles-1)/a){
				printf("Bubble %i has position (%4.2E, %4.2E), radius %4.2E.\n",
					(int)i,
					solution[save_count].pos[(int)i].x,
					solution[save_count].pos[(int)i].y,
					solution[save_count].R_t[(int)i]);
				}
			}
			printf("resimax = %4.2E\n\n",resimax);
			if(debug->fg){
				printf("fg Grid\n");
				for (double j = 0; j <= j2m - 1; j+= (j2m-1)/a){
				if (!j){
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("%4.2E\t", solution[save_count].f_g[i2m * (int)j + (int)i]);
				}
				printf("\n");
				}
			}
			if(debug->p0){
				printf("p0 Grid\n");
				for (double j = 0; j <= j1m - 1; j += (j1m - 1)/a){
				if (!j){
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("%4.2E\t", solution[save_count].p0[i1m * (int)j + (int)i]);
				}
				printf("\n");
				}
			}
			if(debug->pxy){
				printf("px Grid\n");
				for (double j = 0; j <= j1m - 1; j += (j1m - 1)/a){
				if (!j){
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("%4.2E\t", solution[save_count].p[i1m * (int)j + (int)i].x);
				}
				printf("\n");
				}
			}
			if(debug->pxy){
				printf("py Grid\n");
				for (double j = 0; j <= j1m - 1; j += (j1m - 1)/a){
				if (!j){
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i1m - 1; i += (i1m - 1)/a){
				printf("%4.2E\t", solution[save_count].p[i1m * (int)j + (int)i].y);
				}
				printf("\n");
				}
			}
			if(debug->T){
				printf("T Grid\n");
				for (double j = 0; j <= j2m - 1; j += (j2m - 1)/a){
				if (!j){
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("%4.2E\t", solution[save_count].T[i2m * (int)j + (int)i]);
				}
				printf("\n");
				}
			}
			if(debug->vxy){
				printf("vx Grid\n");
				for (double j = 0; j <= j2m - 1; j += (j2m - 1)/a){
				if (!j){
				for (double i = 0; i <= i2n - 1; i += (i2n - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i2n - 1; i += (i2n - 1)/a){
				printf("%4.2E\t", solution[save_count].vx[i2n * (int)j + (int)i]);
				}
				printf("\n");
				}
			}
			if(debug->vxy){
				printf("vy Grid\n");
				for (double j = 0; j <= j2n - 1; j += (j2n - 1)/a){
				if (!j){
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("\t(%i)\t",(int)i);
				}
				printf("\n");
				}
				printf("(%i)\t",(int)j);
				for (double i = 0; i <= i2m - 1; i += (i2m - 1)/a){
				printf("%4.2E\t", solution[save_count].vy[i2m * (int)j + (int)i]);
				}
				printf("\n");
				}
			}
			printf("\n\n\n");
		#endif
			save_count++;
		}
	}while(
	(((*sim_params).NSTEPMAX != 0) && (nstep < (*sim_params).NSTEPMAX))
	||
	(((*sim_params).TSTEPMAX != 0) && (tstep < (*sim_params).TSTEPMAX)));

	// Destroy the variables to prevent further errors
	if(destroy_CUDA_variables(bub_params)){
		exit(EXIT_FAILURE);
	}

	return solution;
} // solve_bubbles()

int initialize_CUDA_variables(	grid_t		*grid_size,
				PML_t		*PML,
				sim_params_t	*sim_params,
				plane_wave_t	*plane_wave,
				array_index_t	*array_index,
				mix_params_t	*mix_params,
				bub_params_t	*bub_params){

	j0m  = array_index->jendm  - array_index->jstam + 1;
	j0n  = array_index->jendn  - array_index->jstan + 1;
	i1m = array_index->iend1m - array_index->ista1m + 1;
	j1m = array_index->jend1m - array_index->jsta1m + 1;
	i1n = array_index->iend1n - array_index->ista1n + 1;
	j1n = array_index->jend1n - array_index->jsta1n + 1;
	i2m = array_index->iend2m - array_index->ista2m + 1;
	j2m = array_index->jend2m - array_index->jsta2m + 1;
	i2n = array_index->iend2n - array_index->ista2n + 1;
	j2n = array_index->jend2n - array_index->jsta2n + 1;

	m1Vol =  i1m * j1m;
	m2Vol =  i2m * j2m;
	v_xVol = i2n * j2m;
	v_yVol = i2m * j2n;
	E_xVol = i1n * j1m;
	E_yVol = i1m * j1n;



	cudaMalloc(	(void **)&mixture_htod,		sizeof(mixture_t));
	cudaMallocPitch((void **)&mixture_htod.T, 	&T_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.vx, 	&vx_pitch,	sizeof(double)*i2n, j2m);
	cudaMallocPitch((void **)&mixture_htod.vy, 	&vy_pitch,	sizeof(double)*i2m, j2n);
	cudaMallocPitch((void **)&mixture_htod.c_sl, 	&c_sl_pitch,	sizeof(double)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.rho_m, 	&rho_m_pitch,	sizeof(double)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.rho_l, 	&rho_l_pitch,	sizeof(double)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.f_g, 	&f_g_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.f_gn, 	&f_gn_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.f_gm, 	&f_gm_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.k_m, 	&k_m_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.C_pm, 	&C_pm_pitch,	sizeof(double)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.Work, 	&Work_pitch,	sizeof(double)*i2m, j2m);
	cudaMallocPitch((void **)&mixture_htod.Ex, 	&Ex_pitch,	sizeof(double)*i1n, j1m);
	cudaMallocPitch((void **)&mixture_htod.Ey, 	&Ey_pitch,	sizeof(double)*i1m, j1n);
	cudaMallocPitch((void **)&mixture_htod.p0, 	&p0_pitch,	sizeof(double)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.p, 	&p_pitch,	sizeof(double2)*i1m, j1m);
	cudaMallocPitch((void **)&mixture_htod.pn, 	&pn_pitch,	sizeof(double2)*i1m, j1m);

	#ifdef _DEBUG_
		printf("T_pitch = %i\n", (int)T_pitch);
		printf("vx_pitch = %i\n", (int)vx_pitch);
		printf("vy_pitch = %i\n", (int)vy_pitch);
		printf("c_sl_pitch = %i\n", (int)c_sl_pitch);
		printf("rho_m_pitch = %i\n", (int)rho_m_pitch);
		printf("rho_l_pitch = %i\n", (int)rho_l_pitch);
		printf("f_g_pitch = %i\n", (int)f_g_pitch);
		printf("f_gn_pitch = %i\n", (int)f_gn_pitch);
		printf("f_gm_pitch = %i\n", (int)f_gm_pitch);
		printf("k_m_pitch = %i\n", (int)k_m_pitch);
		printf("C_pm_pitch = %i\n", (int)C_pm_pitch);
		printf("Work_pitch = %i\n", (int)Work_pitch);
		printf("Ex_pitch = %i\n", (int)Ex_pitch);
		printf("Ey_pitch = %i\n", (int)Ey_pitch);
		printf("p0_pitch = %i\n", (int)p0_pitch);
		printf("p_pitch = %i\n", (int)p_pitch);
		printf("pn_pitch = %i\n", (int)pn_pitch);
	#endif

	if(bub_params->enabled){
		cudaMalloc((void **)	&bubbles_htod,		sizeof(bubble_t));
		cudaMalloc((void **)	&bubbles_htod.ibm,	sizeof(int2)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.ibn,	sizeof(int2)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.pos,	sizeof(double2)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.R_t,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.R_p,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.R_pn,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.R_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.R_nn,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.d1_R_p,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.d1_R_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.PG_p,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.PG_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.PL_p,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.PL_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.PL_m,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.Q_B,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.n_B,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.dt,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.dt_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.re,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.re_n,	sizeof(double)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.v_B,	sizeof(double2)*numBubbles);
		cudaMalloc((void **)	&bubbles_htod.v_L,	sizeof(double2)*numBubbles);
	}

	cudaMalloc((void **)	&sigma_htod,		sizeof(sigma_t));
	cudaMalloc((void **)	&sigma_htod.mx,		sizeof(double)*sigma_h.mx_size);
	cudaMalloc((void **)	&sigma_htod.my,		sizeof(double)*sigma_h.my_size);
	cudaMalloc((void **)	&sigma_htod.nx,		sizeof(double)*sigma_h.nx_size);
	cudaMalloc((void **)	&sigma_htod.ny,		sizeof(double)*sigma_h.ny_size);

	cudaMalloc((void **)	&grid_htod,		sizeof(grid_gen));
	cudaMalloc((void **)	&grid_htod.xu,		sizeof(double)*grid_h.xu_size);
	cudaMalloc((void **)	&grid_htod.rxp,		sizeof(double)*grid_h.rxp_size);;

	checkCUDAError("Memory Allocation");

//	cudaMemcpy(&mixture_htod,	&mixture_h,
//			sizeof(mixture_t),	cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.T, T_pitch,
			mixture_h.T, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	checkCUDAError("T To Device");
	cudaMemcpy2D(	mixture_htod.p0, p0_pitch,
			mixture_h.p0, sizeof(double)*i1m, sizeof(double)*i1m, j1m,
			cudaMemcpyHostToDevice);
	checkCUDAError("p0 To Device");
	cudaMemcpy2D(	mixture_htod.p, p_pitch,
			mixture_h.p, sizeof(double2)*i1m, sizeof(double2)*i1m, j1m,
			cudaMemcpyHostToDevice);
	checkCUDAError("p To Device");
	cudaMemcpy2D(	mixture_htod.pn, pn_pitch,
			mixture_h.pn, sizeof(double2)*i1m, sizeof(double2)*i1m, j1m,
			cudaMemcpyHostToDevice);
	checkCUDAError("pn To Device");
	cudaMemcpy2D(	mixture_htod.vx, vx_pitch,
			mixture_h.vx, sizeof(double)*i2n, sizeof(double)*i2n, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.vy, vy_pitch,
			mixture_h.vy, sizeof(double)*i2m, sizeof(double)*i2m, j2n,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.c_sl, c_sl_pitch,
			mixture_h.c_sl, sizeof(double)*i1m, sizeof(double)*i1m, j1m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.rho_m, rho_m_pitch,
			mixture_h.rho_m, sizeof(double)*i1m, sizeof(double)*i1m, j1m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.rho_l, rho_l_pitch,
			mixture_h.rho_l, sizeof(double)*i1m, sizeof(double)*i1m, j1m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.f_g, f_g_pitch,
			mixture_h.f_g, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.f_gn, f_gn_pitch,
			mixture_h.f_gn, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.f_gm, f_gm_pitch,
			mixture_h.f_gm, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.k_m, k_m_pitch,
			mixture_h.k_m, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.C_pm, C_pm_pitch,
			mixture_h.C_pm, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.Work, Work_pitch,
			mixture_h.Work, sizeof(double)*i2m, sizeof(double)*i2m, j2m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.Ex, Ex_pitch,
			mixture_h.Ex, sizeof(double)*i1n, sizeof(double)*i1n, j1m,
			cudaMemcpyHostToDevice);
	cudaMemcpy2D(	mixture_htod.Ey, Ey_pitch,
			mixture_h.Ey, sizeof(double)*i1m, sizeof(double)*i1m, j1n,
			cudaMemcpyHostToDevice);


	checkCUDAError("Mixture To Device");
	if(bub_params->enabled){
		cudaMemcpy(bubbles_htod.ibm,	bubbles_h.ibm,
				sizeof(int2)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.ibn,	bubbles_h.ibn,
				sizeof(int2)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.pos,	bubbles_h.pos,
				sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.R_t,	bubbles_h.R_t,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.R_p,	bubbles_h.R_p,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.R_pn,	bubbles_h.R_pn,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.R_n,	bubbles_h.R_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.R_nn,	bubbles_h.R_nn,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.d1_R_p,	bubbles_h.d1_R_p,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.d1_R_n,	bubbles_h.d1_R_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.PG_p,	bubbles_h.PG_p,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.PG_n,	bubbles_h.PG_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.PL_p,	bubbles_h.PL_p,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.PL_n,	bubbles_h.PL_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.PL_m,	bubbles_h.PL_m,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.Q_B,	bubbles_h.Q_B,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.n_B,	bubbles_h.n_B,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.dt,	bubbles_h.dt,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.dt_n,	bubbles_h.dt_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.re,	bubbles_h.re,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.re_n,	bubbles_h.re_n,
				sizeof(double)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.v_B,	bubbles_h.v_B,
				sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
		cudaMemcpy(bubbles_htod.v_L,	bubbles_h.v_L,
				sizeof(double2)*numBubbles,	cudaMemcpyHostToDevice);
	}


//	cudaMemcpy(&sigma_htod,		&sigma_h,
//			sizeof(sigma_t),		cudaMemcpyHostToDevice);

	cudaMemcpy(sigma_htod.mx,	sigma_h.mx,
			sizeof(double)*sigma_h.mx_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(sigma_htod.my,	sigma_h.my,
			sizeof(double)*sigma_h.my_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(sigma_htod.nx,	sigma_h.nx,
			sizeof(double)*sigma_h.nx_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(sigma_htod.ny,	sigma_h.ny,
			sizeof(double)*sigma_h.ny_size,	cudaMemcpyHostToDevice);
	checkCUDAError("Sigma To Device");

//	cudaMemcpy(&grid_htod,		&grid_h,
//			sizeof(grid_gen),		cudaMemcpyHostToDevice);



	cudaMemcpy(grid_htod.xu,	grid_h.xu,
			sizeof(double)*grid_h.xu_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(grid_htod.rxp,	grid_h.rxp,
			sizeof(double)*grid_h.rxp_size,	cudaMemcpyHostToDevice);
	checkCUDAError("Grid To Device");

	// Throw constants into cache
	double3 tmp;

	tmp.x = 1.0/((double)sim_params->deltaBand);
	tmp.y = 2.0 * Pi / ((double)sim_params->deltaBand) * grid_size->rdx;
	tmp.z = 2.0 * Pi / ((double)sim_params->deltaBand) * grid_size->rdy;

	cutilSafeCall(cudaMemcpyToSymbol(mixture_c,	&mixture_htod,	sizeof(mixture_t)));

	cutilSafeCall(cudaMemcpyToSymbol(sigma_c, &sigma_htod, sizeof(sigma_t)));
	cutilSafeCall(cudaMemcpyToSymbol(gridgen_c, &grid_htod, sizeof(grid_gen)));

	cutilSafeCall(cudaMemcpyToSymbol(Pi, &Pi_h, sizeof(double)));
	cutilSafeCall(cudaMemcpyToSymbol(Pi4r3, &Pi4r3_h, sizeof(double)));
	cutilSafeCall(cudaMemcpyToSymbol(delta_coef, &tmp, sizeof(double3)));
	cutilSafeCall(cudaMemcpyToSymbol(array_c, array_index, sizeof(array_index_t)));
	cutilSafeCall(cudaMemcpyToSymbol(grid_c, grid_size, sizeof(grid_t)));
	cutilSafeCall(cudaMemcpyToSymbol(sim_params_c, sim_params, sizeof(sim_params_t)));
	cutilSafeCall(cudaMemcpyToSymbol(plane_wave_c, plane_wave, sizeof(plane_wave_t)));
	cutilSafeCall(cudaMemcpyToSymbol(mix_params_c, mix_params, sizeof(mix_params_t)));
	cutilSafeCall(cudaMemcpyToSymbol(PML_c, PML, sizeof(PML_t)));

	if(bub_params->enabled){
		cutilSafeCall(cudaMemcpyToSymbol(bubbles_c,	&bubbles_htod,	sizeof(bubble_t)));
		cutilSafeCall(cudaMemcpyToSymbol(bub_params_c, bub_params, sizeof(bub_params_t)));
		cutilSafeCall(cudaMemcpyToSymbol(num_bubbles, &numBubbles, sizeof(int)));
	}


	checkCUDAError("Constant Memory Cache");

	// Determine the required CUDA parameters

	T_width		= T_pitch 	/ sizeof(double);
	P_width		= P_pitch	/ sizeof(double);
	p0_width	= p0_pitch 	/ sizeof(double);
	p_width		= p_pitch	/ sizeof(double2);
	pn_width	= pn_pitch	/ sizeof(double2);
	vx_width	= vx_pitch	/ sizeof(double);
	vy_width	= vy_pitch	/ sizeof(double);
	c_sl_width	= c_sl_pitch	/ sizeof(double);
	rho_m_width	= rho_m_pitch	/ sizeof(double);
	rho_l_width	= rho_l_pitch	/ sizeof(double);
	f_g_width	= f_g_pitch	/ sizeof(double);
	f_gn_width	= f_gn_pitch	/ sizeof(double);
	f_gm_width	= f_gm_pitch	/ sizeof(double);
	k_m_width	= k_m_pitch	/ sizeof(double);
	C_pm_width	= C_pm_pitch	/ sizeof(double);
	Work_width	= Work_pitch	/ sizeof(double);
	Ex_width	= Ex_pitch	/ sizeof(double);
	Ey_width	= Ey_pitch	/ sizeof(double);

	return 0;
}

int destroy_CUDA_variables(bub_params_t *bub_params){

	cutilSafeCall(cudaFree(mixture_htod.T));
	cutilSafeCall(cudaFree(mixture_htod.vx));
	cutilSafeCall(cudaFree(mixture_htod.vy));
	cutilSafeCall(cudaFree(mixture_htod.c_sl));
	cutilSafeCall(cudaFree(mixture_htod.rho_m));
	cutilSafeCall(cudaFree(mixture_htod.rho_l));
	cutilSafeCall(cudaFree(mixture_htod.f_g));
	cutilSafeCall(cudaFree(mixture_htod.f_gn));
	cutilSafeCall(cudaFree(mixture_htod.f_gm));
	cutilSafeCall(cudaFree(mixture_htod.k_m));
	cutilSafeCall(cudaFree(mixture_htod.C_pm));
	cutilSafeCall(cudaFree(mixture_htod.Work));
	cutilSafeCall(cudaFree(mixture_htod.Ex));
	cutilSafeCall(cudaFree(mixture_htod.Ey));
	cutilSafeCall(cudaFree(mixture_htod.p0));
	cutilSafeCall(cudaFree(mixture_htod.p));
	cutilSafeCall(cudaFree(mixture_htod.pn));

	if(bub_params->enabled){
		cutilSafeCall(cudaFree(bubbles_htod.ibm));
		cutilSafeCall(cudaFree(bubbles_htod.ibn));
		cutilSafeCall(cudaFree(bubbles_htod.pos));
		cutilSafeCall(cudaFree(bubbles_htod.R_t));
		cutilSafeCall(cudaFree(bubbles_htod.R_p));
		cutilSafeCall(cudaFree(bubbles_htod.R_pn));
		cutilSafeCall(cudaFree(bubbles_htod.R_n));
		cutilSafeCall(cudaFree(bubbles_htod.R_nn));
		cutilSafeCall(cudaFree(bubbles_htod.d1_R_p));
		cutilSafeCall(cudaFree(bubbles_htod.d1_R_n));
		cutilSafeCall(cudaFree(bubbles_htod.PG_p));
		cutilSafeCall(cudaFree(bubbles_htod.PG_n));
		cutilSafeCall(cudaFree(bubbles_htod.PL_p));
		cutilSafeCall(cudaFree(bubbles_htod.PL_n));
		cutilSafeCall(cudaFree(bubbles_htod.PL_m));
		cutilSafeCall(cudaFree(bubbles_htod.Q_B));
		cutilSafeCall(cudaFree(bubbles_htod.n_B));
		cutilSafeCall(cudaFree(bubbles_htod.dt));
		cutilSafeCall(cudaFree(bubbles_htod.dt_n));
		cutilSafeCall(cudaFree(bubbles_htod.re));
		cutilSafeCall(cudaFree(bubbles_htod.re_n));
		cutilSafeCall(cudaFree(bubbles_htod.v_B));
		cutilSafeCall(cudaFree(bubbles_htod.v_L));
	}

	cutilSafeCall(cudaFree(sigma_htod.mx));
	cutilSafeCall(cudaFree(sigma_htod.my));
	cutilSafeCall(cudaFree(sigma_htod.nx));
	cutilSafeCall(cudaFree(sigma_htod.ny));

	cutilSafeCall(cudaFree(grid_htod.xu));
	cutilSafeCall(cudaFree(grid_htod.rxp));


	checkCUDAError("Memory Allocation");
	return 0;
}

int initialize_variables(	grid_t		*grid_size,
				PML_t		*PML,
				sim_params_t	*sim_params,
				plane_wave_t	*plane_wave,
				array_index_t	*array_index,
				mix_params_t	*mix_params,
				bub_params_t	*bub_params){
*grid_size = init_grid_size(*grid_size);

	// Plane Wave
	*plane_wave = init_plane_wave(*plane_wave, *grid_size);

	// Array index

	*array_index = init_array(*grid_size, *sim_params);

	// Mixture parameters

	*mix_params = init_mix();

	mix_params->dt = mix_set_time_increment(*sim_params, min((*grid_size).dx,(*grid_size).dy), (*mix_params).cs_inf);

	// Mixture
	mixture_h = init_mix_array(mix_params, *array_index);

	// Sigma for PML
	sigma_h = init_sigma(*PML, *sim_params, *grid_size, *array_index);

	// rxp and xu
	grid_h = init_grid_vector (*array_index, *grid_size);

	if(bub_params->enabled){
		// Fill in missing bubble parameters
		*bub_params = init_bub_params(*bub_params, *sim_params, (*mix_params).dt);

		// Bubble
		bubbles_h = init_bub_array(bub_params, mix_params, array_index, grid_size, plane_wave);
	}

	return 0;
}

grid_t init_grid_size(grid_t grid_size){
	grid_size.dx = (double)grid_size.LX / (double)grid_size.X;
	grid_size.dy = (double)grid_size.LY / (double)grid_size.Y;
	grid_size.rdx = (double) 1.0 / (double)grid_size.dx;
	grid_size.rdy = (double) 1.0 / (double)grid_size.dy;

#ifdef	_DEBUG_
	printf("Grid Size Parameters\n");
	printf("dx = %E\tdy = %E\trdx = %E\trdy = %E\n\n",
		grid_size.dx,
		grid_size.dy,
		grid_size.rdx,
		grid_size.rdy);
#endif
	return grid_size;
}

plane_wave_t init_plane_wave(plane_wave_t plane_wave, grid_t grid_size){
	if (plane_wave.f_dist){
		plane_wave.fp.x = 0.0;
		plane_wave.fp.y = plane_wave.f_dist * 0.5 * sqrt(3.0);
	}
	else{
		plane_wave.fp.x = 0.0;
		plane_wave.fp.y = grid_size.LY * 0.5;
	}
	plane_wave.omega = 2.0 * acos(-1.0) * plane_wave.freq;
	return plane_wave;
}

// Initializes the array index
array_index_t init_array(	const grid_t grid_size,
				const sim_params_t sim_params){

	array_index_t a;

//	a.lmax 	= (sim_params.deltaBand+1)*(sim_params.deltaBand+1);
	a.ms 	= -sim_params.order/2 + 1;
	a.me 	= sim_params.order + a.ms - 1;
	a.ns 	= -sim_params.order/2;
	a.ne 	= sim_params.order + a.ns - 1;

	a.istam	= 1;
	a.iendm	= grid_size.X;
	a.istan	= a.istam - 1;
	a.iendn	= a.iendm;
	a.ista1m = a.istan  + a.ms;
	a.iend1m = a.iendn  + a.me;
	a.ista1n = a.istam  + a.ns;
	a.iend1n = a.iendm  + a.ne;
	a.ista2m = a.ista1n + a.ms;
	a.iend2m = a.iend1n + a.me;
	a.ista2n = a.ista1m + a.ns;
	a.iend2n = a.iend1m + a.ne;

	a.jstam	= 1;
	a.jendm	= grid_size.Y;
	a.jstan	= a.jstam - 1;
	a.jendn	= a.jendm;
	a.jsta1m = a.jstan  + a.ms;
	a.jend1m = a.jendn  + a.me;
	a.jsta1n = a.jstam  + a.ns;
	a.jend1n = a.jendm  + a.ne;
	a.jsta2m = a.jsta1n + a.ms;
	a.jend2m = a.jend1n + a.me;
	a.jsta2n = a.jsta1m + a.ns;
	a.jend2n = a.jend1m + a.ne;

#ifdef _DEBUG_
	printf("Array Index\n");
//	printf("lmax : %i\n", a.lmax);
	printf("ms : %i\t", a.ms);
	printf("me : %i\n", a.me);
	printf("ns : %i\t", a.ns);
	printf("ne : %i\n\n", a.ne);
	printf("istam : %i\t", a.istam);
	printf("iendm : %i\t\t", a.iendm);
	printf("istan : %i\t", a.istan);
	printf("iendn : %i\n", a.iendn);
	printf("jstam : %i\t", a.jstam);
	printf("jendm : %i\t\t", a.jendm);
	printf("jstan : %i\t", a.jstan);
	printf("jendn : %i\n", a.jendn);
	printf("ista1m : %i\t", a.ista1m);
	printf("iend1m : %i\t\t", a.iend1m);
	printf("ista1n : %i\t", a.ista1n);
	printf("iend1n : %i\n", a.iend1n);
	printf("jsta1m : %i\t", a.jsta1m);
	printf("jend1m : %i\t\t", a.jend1m);
	printf("jsta1n : %i\t", a.jsta1n);
	printf("jend1n : %i\n", a.jend1n);
	printf("ista2m : %i\t", a.ista2m);
	printf("iend2m : %i\t\t", a.iend2m);
	printf("ista2n : %i\t", a.ista2n);
	printf("iend2n : %i\n", a.iend2n);
	printf("jsta2m : %i\t", a.jsta2m);
	printf("jend2m : %i\t\t", a.jend2m);
	printf("jsta2n : %i\t", a.jsta2n);
	printf("jend2n : %i\n\n", a.jend2n);
#endif //_DEBUG_

	return a;
} // init_array()

// Initializes mixture parameters
mix_params_t init_mix(){
	mix_params_t mix_params;
	mix_params.T_inf	= 293.15;
	mix_params.P_inf	= 0.1e6;
	mix_params.fg_inf	= 1.0e-7;
	mix_params.rho_inf	= density_water(mix_params.P_inf,mix_params.T_inf);
	mix_params.cs_inf	= adiabatic_sound_speed_water(mix_params.P_inf,mix_params.T_inf);

	return mix_params;
} // init_mix()

// Set the mixture time step
double mix_set_time_increment(sim_params_t sim_params, double dx_min, double u_max){
#ifdef _DEBUG_
	printf("sim_params.cfl = %E\tdx_min = %E\tu_max = %E\n",sim_params.cfl, dx_min, u_max);
	printf("dt = %E\n\n",sim_params.cfl * dx_min / u_max);
#endif //_DEBUG_
	return sim_params.cfl * dx_min / u_max;
} // mix_set_time_increment()

// Initializes implicit bubble parameters
bub_params_t init_bub_params(	bub_params_t bub_params, sim_params_t sim_params, double dt0){

	bub_params.R03		= bub_params.R0 * bub_params.R0 * bub_params.R0;
	bub_params.PG0		= bub_params.PL0 + 2.0 * bub_params.sig/bub_params.R0;
	bub_params.coeff_alpha	= bub_params.gam * bub_params.PG0 * bub_params.R03
				/ (2.0 * (bub_params.gam - 1.0) * bub_params.T0 * bub_params.K0);
	bub_params.dt0		= 0.1 * dt0;
	bub_params.npi		= 0;
	bub_params.mbs		= -sim_params.deltaBand / 2 + 1;
	bub_params.mbe		= sim_params.deltaBand + bub_params.mbs - 1;
	bub_params.nbs		= -sim_params.deltaBand / 2;
	bub_params.nbe		= sim_params.deltaBand + bub_params.nbs - 1;
#ifdef _DEBUG_
	printf("Bubble Parameters\n");
	printf("PG0 = %E\tdt0 = %E\nmbs = %i\tmbe = %i\tnbs = %i\tnbe = %i\n\n",
		bub_params.PG0,
		bub_params.dt0,
		bub_params.mbs,
		bub_params.mbe,
		bub_params.nbs,
		bub_params.nbe);
#endif // _DEBUG_
	return bub_params;
} // init_bub_params()

// Initializes useful index variables
grid_gen init_grid_vector (	array_index_t array_index,
				grid_t grid_size){
	grid_gen grid;

	grid.rxp = (double*) calloc((array_index.iend2m - array_index.ista2m + 1), sizeof(double));
	grid.xu  = (double*) calloc((array_index.iend2n - array_index.ista2n + 1), sizeof(double));

	grid.rxp_size = (array_index.iend2m - array_index.ista2m + 1);
	grid.xu_size = (array_index.iend2n - array_index.ista2n + 1);

	for (int i = array_index.ista2m; i <= array_index.iend2m; i++){
		grid.rxp[i - array_index.ista2m] = 1.0/(grid_size.dx * ((double)i - 0.5));

	}
	for (int i = array_index.ista2n; i <= array_index.iend2n; i++){
		grid.xu[i - array_index.ista2n] = ((double)i) * grid_size.dx;

	}

	return grid;
} // init_grid_vector()

// Initialize the mixture array
mixture_t init_mix_array(mix_params_t * mix_params, array_index_t array_index){
	mixture_t mix;

#ifdef _DEBUG_
	printf("Mixing mixture...\n");
#endif

	int m1Vol = (array_index.iend1m - array_index.ista1m + 1) * (array_index.jend1m - array_index.jsta1m + 1);
	int m2Vol = (array_index.iend2m - array_index.ista2m + 1) * (array_index.jend2m - array_index.jsta2m + 1);
	int v_xVol = (array_index.iend2n - array_index.ista2n + 1) * (array_index.jend2m - array_index.jsta2m + 1);
	int v_yVol = (array_index.iend2m - array_index.ista2m + 1) * (array_index.jend2n - array_index.jsta2n + 1);
	int E_xVol = (array_index.iend1n - array_index.ista1n + 1) * (array_index.jend1m - array_index.jsta1m + 1);
	int E_yVol = (array_index.iend1m - array_index.ista1m + 1) * (array_index.jend1n - array_index.jsta1n + 1);

	mix.T 		= (double*) calloc(m2Vol,  sizeof(double));
	mix.p0		= (double*) calloc(m1Vol,  sizeof(double));
	mix.p		= (double2*) calloc(m1Vol,  sizeof(double2));
	mix.pn		= (double2*) calloc(m1Vol,  sizeof(double2));
	mix.c_sl	= (double*) calloc(m1Vol,  sizeof(double));
	mix.rho_m	= (double*) calloc(m1Vol,  sizeof(double));
	mix.rho_l	= (double*) calloc(m1Vol,  sizeof(double));
	mix.f_g		= (double*) calloc(m2Vol,  sizeof(double));
	mix.f_gn	= (double*) calloc(m2Vol,  sizeof(double));
	mix.f_gm	= (double*) calloc(m2Vol,  sizeof(double));
	mix.k_m		= (double*) calloc(m2Vol,  sizeof(double));
	mix.C_pm	= (double*) calloc(m1Vol,  sizeof(double));
	mix.Work	= (double*) calloc(m2Vol,  sizeof(double));
	mix.vx		= (double*) calloc(v_xVol,  sizeof(double));
	mix.vy		= (double*) calloc(v_yVol,  sizeof(double));
	mix.Ex		= (double*) calloc(E_xVol,  sizeof(double));
	mix.Ey		= (double*) calloc(E_yVol,  sizeof(double));

	for (int i = 0; i < m1Vol; i++){
		mix.p0[i]	= 0.0;
		mix.p[i]	= make_double2(0.0, 0.0);
		mix.pn[i]	= make_double2(0.0, 0.0);
		mix.rho_m[i]	= mix.rho_l[i] = density_water(mix_params->P_inf, mix_params->T_inf);

		mix.c_sl[i] = adiabatic_sound_speed_water(mix_params->P_inf, mix_params->T_inf);

		mix.C_pm[i] = specific_heat_water(mix_params->T_inf);
		mix.k_m[i] = thermal_conductivity_water(mix_params->T_inf);
	}
	for (int i = 0; i < v_xVol; i++){
		mix.vx[i] = 0.0;
	}
	for (int i = 0; i < v_yVol; i++){
		mix.vy[i] = 0.0;
	}
	for (int i = 0; i < E_xVol; i++){
		mix.Ex[i] = 0.0;
	}
	for (int i = 0; i < E_yVol; i++){
		mix.Ey[i] = 0.0;
	}
	for (int i = 0; i < m2Vol; i++){
		mix.T[i] = 0.0;
		mix.f_g[i] = 0.0;//(double) i/m2Vol;
		mix.Work[i] = 0;
	}
	printf("Mixture grid generated.\n\n");
	return mix;
} // init_mix_array()

bubble_t init_bub_array(bub_params_t *bub_params, mix_params_t *mix_params, array_index_t *array_index, grid_t *grid_size, plane_wave_t *plane_wave){
	double2 pos = make_double2(0.0, 0.0);
	host_vector<bubble_t_aos> bub;
	bubble_t_aos init_bubble;
	bubble_t ret_bub;

#ifdef _DEBUG_
	printf("Baking bubbles...\n");
#endif
	for (int i = (*array_index).istam; i <= (*array_index).iendm; i++){
		pos.x = ( (double)i - 0.5) * (*grid_size).dx;
		for (int j = (*array_index).jstam; j <= (*array_index).jendm; j++){
			pos.y = ( (double)j - 0.5) * (*grid_size).dy;

			if ((*plane_wave).box_size){
				if((abs(pos.x - (*plane_wave).fp.x) < 0.5 * (*plane_wave).box_size)  &&(abs(pos.y - (*plane_wave).fp.y) < 0.5 * (*plane_wave).box_size) ){
					init_bubble = bubble_input(	pos,
									(*bub_params).fg0,
									*bub_params,
									*grid_size,
									*plane_wave);
					bub.push_back(init_bubble);
				}
			}
			else{
				init_bubble = bubble_input(	pos,
								(*bub_params).fg0,
								*bub_params,
								*grid_size,
								*plane_wave);
				bub.push_back(init_bubble);
			}
		}
	}
	numBubbles = (*bub_params).npi = bub.size();

	ret_bub.ibm	= (int2*) calloc(bub.size(),  sizeof(int2));
	ret_bub.ibn	= (int2*) calloc(bub.size(),  sizeof(int2));
	ret_bub.pos	= (double2*) calloc(bub.size(),  sizeof(double2));
	ret_bub.R_t	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.R_p	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.R_pn	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.R_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.R_nn	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.d1_R_p	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.d1_R_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.PG_p	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.PG_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.PL_p	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.PL_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.PL_m	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.Q_B	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.n_B	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.dt	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.dt_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.re	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.re_n	= (double*) calloc(bub.size(),  sizeof(double));
	ret_bub.v_B	= (double2*) calloc(bub.size(),  sizeof(double2));
	ret_bub.v_L	= (double2*) calloc(bub.size(),  sizeof(double2));

	for (int i = 0; i < (*bub_params).npi; i++){
		ret_bub.ibm[i]		= bub[i].ibm;
		ret_bub.ibn[i]		= bub[i].ibn;
		ret_bub.pos[i]		= bub[i].pos;
		ret_bub.R_t[i]		= bub[i].R_t;
		ret_bub.R_p[i]		= bub[i].R_p;
		ret_bub.R_pn[i]		= bub[i].R_pn;
		ret_bub.R_n[i]		= bub[i].R_n;
		ret_bub.R_nn[i]		= bub[i].R_nn;
		ret_bub.d1_R_p[i]	= bub[i].d1_R_p;
		ret_bub.d1_R_n[i]	= bub[i].d1_R_n;
		ret_bub.PG_p[i]		= bub[i].PG_p;
		ret_bub.PG_n[i]		= bub[i].PG_n;
		ret_bub.PL_p[i]		= bub[i].PL_p;
		ret_bub.PL_n[i]		= bub[i].PL_n;
		ret_bub.PL_m[i]		= bub[i].PL_m;
		ret_bub.Q_B[i]		= bub[i].Q_B;
		ret_bub.n_B[i]		= bub[i].n_B;
		ret_bub.dt[i]		= bub[i].dt;
		ret_bub.dt_n[i]		= bub[i].dt_n;
		ret_bub.re[i]		= bub[i].re;
		ret_bub.re_n[i]		= bub[i].re_n;
		ret_bub.v_B[i]		= bub[i].v_B;
		ret_bub.v_L[i]		= bub[i].v_L;
	}
#ifdef _DEBUG_
	printf("%i bubbles initialized.\n\n", (*bub_params).npi);
#endif // _DEBUG_
	return ret_bub;
}

bubble_t_aos bubble_input(double2 pos, double fg_in, bub_params_t bub_params, grid_t grid_size, plane_wave_t plane_wave){
	bubble_t_aos new_bubble;

	double Pi = acos(-1.0);

	new_bubble.pos 	= pos;

	new_bubble.R_t 	= bub_params.R0;
	new_bubble.R_p	= new_bubble.R_pn	= bub_params.R0;
	new_bubble.R_n	= new_bubble.R_nn	= bub_params.R0;

	new_bubble.d1_R_p	= new_bubble.d1_R_n	= 0.0;

	new_bubble.PG_p	= new_bubble.PG_n	= bub_params.PG0;

	new_bubble.PL_p	= new_bubble.PL_n	= new_bubble.PL_m	= 0.0;

	if(plane_wave.cylindrical){
		new_bubble.n_B	= fg_in * (pos.x * grid_size.dx * grid_size.dy)
			/ (4.0 / 3.0 * Pi * pow(new_bubble.R_t,3));
	}
	else{
		new_bubble.n_B	= fg_in * (grid_size.dx * grid_size.dy)
			/ (4.0 / 3.0 * Pi * pow(new_bubble.R_t,3));
	}

	new_bubble.Q_B	= 0.0;

	new_bubble.dt	= new_bubble.dt_n	= bub_params.dt0;
	new_bubble.re	= new_bubble.re_n	= 0.0;
	new_bubble.v_B	= new_bubble.v_L		= make_double2(0.0, 0.0);

	new_bubble.ibm	= make_int2(0,0);
	new_bubble.ibn	= make_int2(0,0);

	return new_bubble;
}

// Initializes the sigma field used for PML
sigma_t init_sigma (	const PML_t PML,
					const sim_params_t sim_params,
					const grid_t grid_size,
					const array_index_t array_index){
#ifdef _DEBUG_
	printf("Generating a perfectly matched layer.\n");
#endif
	sigma_t sigma;

	sigma.mx = (double*) calloc((array_index.iend1m - array_index.ista1m + 1), sizeof(double));
	sigma.my = (double*) calloc((array_index.jend1m - array_index.jsta1m + 1), sizeof(double));
	sigma.nx = (double*) calloc((array_index.iend2n - array_index.ista2n + 1), sizeof(double));
	sigma.ny = (double*) calloc((array_index.jend2n - array_index.jsta2n + 1), sizeof(double));

	sigma.mx_size = (array_index.iend1m - array_index.ista1m + 1);
	sigma.my_size = (array_index.jend1m - array_index.jsta1m + 1);
	sigma.nx_size = (array_index.iend2n - array_index.ista2n + 1);
	sigma.ny_size = (array_index.jend2n - array_index.jsta2n + 1);

	int 	n;
	int 	itmps, itmpe, jtmps, jtmpe;
	double	sigma_x_max, sigma_y_max;
	int	npml = PML.NPML;
	double	sig = PML.sigma;
	double	order	=	PML.order;
	double	dx	=	grid_size.dx;
	double	dy	=	grid_size.dy;
	int	nx	=	grid_size.X;
	int	ny	=	grid_size.Y;
	int	istam	=	array_index.istam;
	int	iendm	=	array_index.iendm;
	int	jstam	=	array_index.jstam;
	int	jendm	=	array_index.jendm;
	int	istan	=	array_index.istan;
	int	iendn	=	array_index.iendn;
	int	jstan	=	array_index.jstan;
	int	jendn	=	array_index.jendn;
	int	ista1m	=	array_index.ista1m;
//	int	iend1m	=	array_index.iend1m;
	int	jsta1m	=	array_index.jsta1m;
//	int	jend1m	=	array_index.jend1m;
/*	int	ista1n	=	array_index.ista1n;
	int	iend1n	=	array_index.iend1n;
	int	jsta1n	=	array_index.jsta1n;
	int	jend1n	=	array_index.jend1n;*/
//	int	ista2m	=	array_index.ista2m;
//	int	iend2m	=	array_index.iend2m;
//	int	jsta2m	=	array_index.jsta2m;
//	int	jend2m	=	array_index.jend2m;
	int	ista2n	=	array_index.ista2n;
//	int	iend2n	=	array_index.iend2n;
	int	jsta2n	=	array_index.jsta2n;
//	int	jend2n	=	array_index.jend2n;
	int	ms	=	array_index.ms;
	int	me	=	array_index.me;
	int	ns	=	array_index.ns;
	int	ne	=	array_index.ne;

	if(PML.X0){
		sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dx * (double)npml);
		if (istam <= npml){
			itmps = max(istam, ms);
			itmpe = min(iendm, npml);
		#ifdef _DEBUG_
			printf("Sigma mx :\t");
		#endif
			for (int i = itmps; i <= itmpe; i++){
				n = npml - i + 1;
				sigma.mx[i - ista1m]	= sigma_x_max * pow(((double)n-0.5)/((double)npml), order);
			#ifdef _DEBUG_
				printf("%4.2E\t",sigma.mx[i-ista1m]);
			#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.X1){
		sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dx * (double)npml);
		if (nx - npml + 1 <= iendm){
			itmps = max(istam, nx - npml + 1);
			itmpe = min(iendm, nx + me);
		#ifdef _DEBUG_
			printf("Sigma mx :\t");
		#endif
			for (int i = itmps; i <= itmpe; i++){
				n = i - nx + npml;
				sigma.mx[i - ista1m]	= sigma_x_max * pow(	((double)n-0.5)
										/((double)npml),
										order);
			#ifdef _DEBUG_
				printf("%4.2E\t",sigma.mx[i-ista1m]);
			#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.Y0){
		sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dy * (double)npml);
		if (jstam <= npml){
			jtmps = max(jstam, ms);
			jtmpe = min(jendm, npml);
		#ifdef _DEBUG_
			printf("Sigma my :\t");
		#endif
			for (int j = jtmps; j <= jtmpe; j++){
				n = npml - j + 1;
				sigma.my[j - jsta1m]	= sigma_y_max * pow(	((double)n-0.5)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.my[j-jsta1m]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.Y1){
		sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dy * (double)npml);
		if (ny - npml + 1 <= jendm){
			jtmps = max(jstam, ny - npml + 1);
			jtmpe = min(jendm, ny + me);
		#ifdef _DEBUG_
			printf("Sigma my :\t");
		#endif
			for (int j = jtmps; j <= jtmpe; j++){
				n = j - ny + npml;
				sigma.my[j - jsta1m]	= sigma_y_max * pow(	((double)n-0.5)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.my[j-jsta1m]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if(PML.X0){
		sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dx * (double)npml);
		if (istan <= npml - 1){
			itmps = max(istan, ms + ns);
			itmpe = min(iendn, npml - 1);
		#ifdef _DEBUG_
			printf("Sigma nx :\t");
		#endif
			for (int i = itmps; i <= itmpe; i++){
				n = npml - i;
				sigma.nx[i - ista2n]	= sigma_x_max * pow(	((double)n)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.nx[i-ista2n]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.X1){
		sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dx * (double)npml);
		if (nx - npml + 1 <= iendn){
			itmps = max(istan, nx - npml + 1);
			itmpe = min(iendn, nx + me + ne + 1);
		#ifdef _DEBUG_
			printf("Sigma nx :\t");
		#endif
			for (int i = itmps; i <= itmpe; i++){
				n = i - nx + npml;
				sigma.nx[i - ista2n]	= sigma_x_max * pow(	((double)n)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.nx[i-ista2n]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.Y0){
		sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dy * (double)npml);
		if (jstan <= npml - 1){
			jtmps = max(jstan, ms + ns);
			jtmpe = min(jendn, npml - 1);
		#ifdef _DEBUG_
			printf("Sigma ny :\t");
		#endif
			for (int j = jtmps; j <= jtmpe; j++){
				n = npml - j;
				sigma.ny[j - jsta2n]	= sigma_y_max * pow(	((double)n)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.ny[j-jsta2n]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	if (PML.Y1){
		sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
				/ (dy * (double)npml);
		if (ny - npml + 1 <= jendn){
			jtmps = max(jstan, ny - npml + 1);
			jtmpe = min(jendn, ny + me + ne + 1);
		#ifdef _DEBUG_
			printf("Sigma ny :\t");
		#endif
			for (int j = jtmps; j <= jtmpe; j++){
				n = j - ny + npml;
				sigma.ny[j - jsta2n]	= sigma_y_max * pow(	((double)n)
										/((double)npml),
										order);
				#ifdef _DEBUG_
					printf("%4.2E\t",sigma.ny[j-jsta2n]);
				#endif
			}
		#ifdef _DEBUG_
			printf("\n");
		#endif
		}
	}
	printf("PML generated.\n\n");
	return sigma;
} // init_sigma()

// Check global memory killswitch, and terminate the program if it's ok
void killSwitch(const bool *ok_d){
	bool *ok = (bool*) calloc(1, sizeof(bool));
	cudaMemcpy(ok, ok_d, sizeof(bool), cudaMemcpyDeviceToHost);
	if (!(*ok)){
		exit(EXIT_FAILURE);
	}
} // killSwitch()

// Checks for any CUDA errors
void checkCUDAError(	const char *msg){
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} // checkCUDAError()
