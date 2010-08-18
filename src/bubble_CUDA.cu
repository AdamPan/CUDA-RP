// CUDA solver for Rayleigh Plesset

#include "bubbles.h"
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

void display(double data[], int xdim, int ydim, int num_lines, char *msg)
{
    printf("%s\n", msg);
    for (double j = 0; j <= ydim - 1; j+= (ydim-1)/num_lines)
    {
        if (!j)
        {
            for (double i = 0; i <= xdim - 1; i += (xdim - 1)/num_lines)
            {
                printf("\t(%i)\t",(int)i);
            }
            printf("\n");
        }
        printf("(%i)\t",(int)j);
        for (double i = 0; i <= i2m - 1; i += (i2m - 1)/num_lines)
        {
            printf("%4.2E\t", data[xdim * (int)j + (int)i]);
        }
        printf("\n");
    }
}

int solve_bubbles(	array_index_t	*array_index,
                   grid_t		*grid_size,
                   PML_t		*PML,
                   sim_params_t	*sim_params,
                   bub_params_t	*bub_params,
                   plane_wave_t	*plane_wave,
                   debug_t		*debug,
                   int 		argc,
                   char 		**argv)
{
    // Variables needed for control structures
    unsigned int nstep = 0;
    double tstep = 0.0, tstepx = 0.0;
    int loop;
    double resimax;
    double s1, s2;
    bool rejoin = 0;

    // Data thread setup
    pthread_t save_thread;
    pthread_attr_t pthread_custom_attr;
    output_plan_t *plan;

    pthread_attr_init(&pthread_custom_attr);

    plan = (output_plan_t *)malloc(sizeof(output_plan_t));

    int max_iter;

#ifdef _DEBUG_
    // Clear terminal for output
    if (system("clear"))
    {
        exit(EXIT_FAILURE);
    }
#endif
    setCUDAflags();

    int num_streams = 3;
    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&stream[i]);
    }
    cudaEvent_t stop[num_streams];
    for (int i = 0; i < num_streams; i++)
    {
        cudaEventCreateWithFlags(&stop[i], cudaEventBlockingSync);
    }

    // Mixture Parameters
    mix_params_t	*mix_params = (mix_params_t*) calloc(1, sizeof(mix_params_t));

    // Initialize Variables
    printf("Computing Simulation Variables...");
    if (initialize_variables(grid_size, PML, sim_params, plane_wave, array_index, mix_params, bub_params))
    {
        exit(EXIT_FAILURE);
    }
    printf("\tdone\n\n");

    printf("Preparing folders...");
    if (initialize_folders())
    {
        exit(EXIT_FAILURE);
    }
    printf("\t\t\tdone\n\n");

    // Allocate memory on the device
    printf("Allocating Memory...");
    if (initialize_CUDA_variables(grid_size, PML, sim_params, plane_wave,array_index, mix_params, bub_params))
    {
        exit(EXIT_FAILURE);
    }
    printf("\t\t\tdone\n\n");

    // Initialization kernels
    printf("Running Initialization Kernels...");
    // Update Bubble Index
    if (bub_params->enabled && update_bubble_indices(stream, stop))
    {
        exit(EXIT_FAILURE);
    }
    // Calculate the initial state void fraction
    if (bub_params->enabled && calculate_void_fraction(mixture_htod, plane_wave, f_g_width, f_g_pitch, stream, stop))
    {
        exit(EXIT_FAILURE);
    }
    // Set f_gn and f_gm to f_g
    if (synchronize_void_fraction(mixture_htod, f_g_pitch, stream, stop))
    {
        exit(EXIT_FAILURE);
    }
    printf("\tdone\n\n");

    // Assign Plan pointers
    plan->mixture_h = mixture_h;
    plan->bubbles_h = bubbles_h;
    plan->array_index = array_index;
    plan->grid_size = grid_size;
    plan->sim_params = sim_params;
    plan->plane_wave = plane_wave;
    plan->debug = debug;

    /************************
     * Main simulation loop	*
     ************************/
    printf("Running the simulation\t\t");
    while (	((sim_params->NSTEPMAX != 0) && (nstep < sim_params->NSTEPMAX)) ||
            ((sim_params->TSTEPMAX != 0) && (tstep < sim_params->TSTEPMAX)))
    {
        nstep++;

        // Accurate time addition
        s1	=	tstep;
        tstep	=	tstep + mix_params->dt + tstepx;
        s2	=	tstep - s1;
        tstepx	=	mix_params->dt + tstepx - s2;
        cudaMemcpyToSymbol(tstep_c, &tstep, sizeof(double));
        checkCUDAError("Set timestamp");

        // Store Bubble and Mixture, and predict void fraction
        if (store_variables(mixture_htod, bubbles_htod, f_g_width, p_width, Work_width, f_g_pitch, Work_pitch, p_pitch, stream, stop))
        {
            exit(EXIT_FAILURE);
        }

        // Update mixture velocity
        if (calculate_velocity_field(vx_width, vy_width, rho_m_width, p0_width, c_sl_width, stream, stop))
        {
            exit(EXIT_FAILURE);
        }

        // Move the bubbles
        if (bub_params->enabled && bubble_motion(bubbles_htod, vx_width, vy_width, stream, stop))
        {
            exit(EXIT_FAILURE);
        }

        // Calculate pressure
        resimax = calculate_pressure_field(mixture_h, mixture_htod, mix_params->P_inf, p0_width, p_width, f_g_width, vx_width, vy_width, rho_l_width, c_sl_width, Work_width, Work_pitch, stream, stop);

        // Subloop for solving Rayleigh Plesset equations
        if (bub_params->enabled)
        {
            loop = 0;
            while (resimax > 1.0e-7f)
            {
                loop++;

                // Find bubble pressure
                if (interpolate_bubble_pressure(p0_width, stream, stop))
                {
                    exit(EXIT_FAILURE);
                }

                // Solve Rayleigh-Plesset Equations
                max_iter = solve_bubble_radii(bubbles_htod, stream, stop);

                // Calculate Void Fraction
                if (calculate_void_fraction(mixture_htod, plane_wave, f_g_width, f_g_pitch, stream, stop))
                {
                    exit(EXIT_FAILURE);
                }

                // Calculate Pressure
                resimax = calculate_pressure_field(mixture_h, mixture_htod, mix_params->P_inf, p0_width, p_width, f_g_width, vx_width, vy_width, rho_l_width, c_sl_width, Work_width, Work_pitch, stream, stop);

#ifdef _DEBUG_
                printf("\033[A\033[2K");
                printf("Simulation step %5i subloop %5i \t resimax = %4.2E, inner loop executed %i times.\n", nstep, loop, resimax, max_iter);
#endif
                if (loop == 10000)
                {
                    printf("Premature Termination at nstep %i, subloop %i\n\n", nstep, loop);
                    break;
                }
            }
        }

        // Calculate mixture temperature
        if (calculate_temperature(mixture_htod, bub_params, k_m_width, T_width, f_g_width, Ex_width, Ey_width, p_width, rho_m_width, C_pm_width, Work_width, Work_pitch, stream, stop))
        {
            exit(EXIT_FAILURE);
        }
        // Calculate mixture properties
        if (calculate_properties(rho_l_width, rho_m_width, c_sl_width, C_pm_width, f_g_width, T_width, stream, stop))
        {
            exit(EXIT_FAILURE);
        }

        // Save data at intervals
        if ((((int)nstep) % ((int)sim_params->DATA_SAVE) == 0))
        {

            // If we're past thre first step, make sure to join with the data thread
#ifdef _OUTPUT_
            if (rejoin)
            {
                pthread_join(save_thread, NULL);
            }
            else
            {
                rejoin = 1;
            }
#endif
            // Copy over requested variables
            if (debug->p0)cudaMemcpy2D(	mixture_h.p0, sizeof(double)*i1m, mixture_htod.p0, p0_pitch, sizeof(double)*i1m, j1m, cudaMemcpyDeviceToHost);
            if (debug->fg)cudaMemcpy2D(	mixture_h.f_g, sizeof(double)*i2m, mixture_htod.f_g, f_g_pitch, sizeof(double)*i2m, j2m, cudaMemcpyDeviceToHost);
            if (debug->T)cudaMemcpy2D(	mixture_h.T, sizeof(double)*i2m, mixture_htod.T, T_pitch, sizeof(double)*i2m, j2m, 	cudaMemcpyDeviceToHost);
            if (debug->vxy)cudaMemcpy2D(	mixture_h.vx, sizeof(double)*i2n, mixture_htod.vx, vx_pitch, sizeof(double)*i2n, j2m, cudaMemcpyDeviceToHost);
            if (debug->vxy)cudaMemcpy2D(	mixture_h.vy, sizeof(double)*i2m, mixture_htod.vy, vy_pitch, sizeof(double)*i2m, j2n, cudaMemcpyDeviceToHost);
            if (debug->bubbles)cudaMemcpy(	bubbles_h.pos, bubbles_htod.pos, sizeof(double2)*numBubbles, cudaMemcpyDeviceToHost);
            if (debug->bubbles)cudaMemcpy(	bubbles_h.R_t, bubbles_htod.R_t, sizeof(double)*numBubbles, cudaMemcpyDeviceToHost);
            if (debug->bubbles)cudaMemcpy(	bubbles_h.PG_p, bubbles_htod.PG_p, sizeof(double)*numBubbles, cudaMemcpyDeviceToHost);

            // Assign the data thread with saving the requested variables
#ifdef _OUTPUT_
            plan->step = nstep;
            pthread_create(&save_thread, &pthread_custom_attr, save_step, (void *)(plan));
#endif

#ifdef _DEBUG_
            if (system("clear"))
            {
                exit(EXIT_FAILURE);
            }
#else
            printf("\r");
#endif

            // Display progress
            if (sim_params->NSTEPMAX != 0)
            {
                printf("Running the simulation...\t\tnstep : %5i / %i", nstep, sim_params->NSTEPMAX);
            }
            else if (sim_params->TSTEPMAX !=0)
            {
                printf("Running the simulation...\t\ttstep : %4.2E / %4.2E", tstep, sim_params->TSTEPMAX);
            }

#ifdef _DEBUG_
            printf("\n");
#else
            fflush(stdout);
#endif

#ifdef _DEBUG_
            // Display the mixture field variables in square grids in the interactive terminal
            double num_lines = debug->display - 1;
            printf("resimax = %4.2E\n\n",resimax);
            if (debug->fg)
            {
                display(mixture_h.fg, i2m, j2m, num_lines, "fg Grid");
            }
            if (debug->p0)
            {
                display(mixture_h.p0, i1m, j1m, num_lines, "p0 Grid");
            }
            if (debug->T)
            {
                display(mixture_h.T, i2m, j2m, num_lines, "T Grid");
            }
            if (debug->vxy)
            {
                display(mixture_h.vx, i2n, j2m, num_lines, "vx Grid");
                display(mixture_h.vy, i2m, j2n, num_lines, "vy Grid");
            }
            printf("\n\n");
#endif
        }
    }

#ifdef _OUTPUT_
    pthread_join(save_thread, NULL);
#endif

#ifndef _DEBUG_
    printf("\r");
#endif
    printf("Running the simulation...\t\tdone                      \n\n");

    printf("Cleaning up...");
    // Destroy the variables to prevent further errors
    if (destroy_CUDA_variables(bub_params))
    {
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

    for (int i = 0; i < num_streams; i++)
    {
        cudaEventDestroy(stop[i]);
    }
    printf("\tdone\n\n");
    return 0;
} // solve_bubbles()

// Initialize simulation variables
int initialize_variables(	grid_t		*grid_size,
                          PML_t		*PML,
                          sim_params_t	*sim_params,
                          plane_wave_t	*plane_wave,
                          array_index_t	*array_index,
                          mix_params_t	*mix_params,
                          bub_params_t	*bub_params)
{
    *grid_size = init_grid_size(*grid_size);

    // Plane Wave
    *plane_wave = init_plane_wave(*plane_wave, *grid_size);

    // Array index
    *array_index = init_array(*grid_size, *sim_params);

    // Sigma for PML
    sigma_h = init_sigma(*PML, *sim_params, *grid_size, *array_index);

    // rxp and xu
    grid_h = init_grid_vector (*array_index, *grid_size);

    // Mixture
    *mix_params = init_mix();
    mix_params->dt = mix_set_time_increment(*sim_params, min(grid_size->dx, grid_size->dy), mix_params->cs_inf);
    mixture_h = init_mix_array(mix_params, *array_index);

    // Bubbles
    if (bub_params->enabled)
    {
        *bub_params = init_bub_params(*bub_params, *sim_params, mix_params->dt);
        bubbles_h = init_bub_array(bub_params, mix_params, array_index, grid_size, plane_wave);
    }

    return 0;
}

// Initialize grid parameters
grid_t init_grid_size(grid_t grid_size)
{
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

// Initialize plane wave coefficients
plane_wave_t init_plane_wave(	plane_wave_t plane_wave,
                              grid_t grid_size)
{
    if (plane_wave.f_dist)
    {
        plane_wave.fp.x = 0.0;
        plane_wave.fp.y = plane_wave.f_dist * 0.5 * sqrt(3.0);
    }
    else
    {
        plane_wave.fp.x = 0.0;
        plane_wave.fp.y = grid_size.LY * 0.5;
    }
    plane_wave.omega = 2.0 * acos(-1.0) * plane_wave.freq;
    return plane_wave;
}

// Initializes the array index
array_index_t init_array(	const grid_t grid_size,
                          const sim_params_t sim_params)
{

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
mix_params_t init_mix()
{
    mix_params_t mix_params;
    mix_params.T_inf	= 293.15;
    mix_params.P_inf	= 0.1e6;
    mix_params.fg_inf	= 1.0e-7;
    mix_params.rho_inf	= density_water(mix_params.P_inf,mix_params.T_inf);
    mix_params.cs_inf	= adiabatic_sound_speed_water(mix_params.P_inf,mix_params.T_inf);

    return mix_params;
} // init_mix()

// Set the mixture time step
double mix_set_time_increment(	sim_params_t sim_params,
                               double dx_min,
                               double u_max)
{
#ifdef _DEBUG_
    printf("sim_params.cfl = %E\tdx_min = %E\tu_max = %E\n",sim_params.cfl, dx_min, u_max);
    printf("dt = %E\n\n",sim_params.cfl * dx_min / u_max);
#endif //_DEBUG_
    return sim_params.cfl * dx_min / u_max;
} // mix_set_time_increment()

// Initializes implicit bubble parameters
bub_params_t init_bub_params(	bub_params_t bub_params,
                              sim_params_t sim_params,
                              double dt0)
{

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
                            grid_t grid_size)
{
    grid_gen grid;

    grid.rxp = (double*) calloc((array_index.iend2m - array_index.ista2m + 1), sizeof(double));
    grid.xu  = (double*) calloc((array_index.iend2n - array_index.ista2n + 1), sizeof(double));

    grid.rxp_size = (array_index.iend2m - array_index.ista2m + 1);
    grid.xu_size = (array_index.iend2n - array_index.ista2n + 1);

    for (int i = array_index.ista2m; i <= array_index.iend2m; i++)
    {
        grid.rxp[i - array_index.ista2m] = 1.0/(grid_size.dx * ((double)i - 0.5));

    }
    for (int i = array_index.ista2n; i <= array_index.iend2n; i++)
    {
        grid.xu[i - array_index.ista2n] = ((double)i) * grid_size.dx;

    }

    return grid;
} // init_grid_vector()

// Initialize the host mixture array
mixture_t init_mix_array(	mix_params_t * mix_params,
                          array_index_t array_index)
{
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

    cudaMallocHost((void**)&mix.T, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.p0, m1Vol*sizeof(double));
    cudaMallocHost((void**)&mix.p, m1Vol*sizeof(double2));
    cudaMallocHost((void**)&mix.pn, m1Vol*sizeof(double2));
    cudaMallocHost((void**)&mix.c_sl, m1Vol*sizeof(double));
    cudaMallocHost((void**)&mix.rho_m, m1Vol*sizeof(double));
    cudaMallocHost((void**)&mix.rho_l, m1Vol*sizeof(double));
    cudaMallocHost((void**)&mix.f_g, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.f_gn, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.f_gm, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.k_m, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.C_pm, m1Vol*sizeof(double));
    cudaMallocHost((void**)&mix.Work, m2Vol*sizeof(double));
    cudaMallocHost((void**)&mix.vx, v_xVol*sizeof(double));
    cudaMallocHost((void**)&mix.vy, v_yVol*sizeof(double));
    cudaMallocHost((void**)&mix.Ex, E_xVol*sizeof(double));
    cudaMallocHost((void**)&mix.Ey, E_yVol*sizeof(double));

    for (int i = 0; i < m1Vol; i++)
    {
        mix.p0[i]	= 0.0;
        mix.p[i]	= make_double2(0.0, 0.0);
        mix.pn[i]	= make_double2(0.0, 0.0);
        mix.rho_m[i]	= mix.rho_l[i] = density_water(mix_params->P_inf, mix_params->T_inf);

        mix.c_sl[i] = adiabatic_sound_speed_water(mix_params->P_inf, mix_params->T_inf);

        mix.C_pm[i] = specific_heat_water(mix_params->T_inf);
        mix.k_m[i] = thermal_conductivity_water(mix_params->T_inf);
    }
    for (int i = 0; i < v_xVol; i++)
    {
        mix.vx[i] = 0.0;
    }
    for (int i = 0; i < v_yVol; i++)
    {
        mix.vy[i] = 0.0;
    }
    for (int i = 0; i < E_xVol; i++)
    {
        mix.Ex[i] = 0.0;
    }
    for (int i = 0; i < E_yVol; i++)
    {
        mix.Ey[i] = 0.0;
    }
    for (int i = 0; i < m2Vol; i++)
    {
        mix.T[i] = 0.0;
        mix.f_g[i] = 0.0;//(double) i/m2Vol;
        mix.Work[i] = 0;
    }
#ifdef _DEBUG_
    printf("Mixture grid generated.\n\n");
#endif
    return mix;
} // init_mix_array()

// Initialize the host bubble array
bubble_t init_bub_array(	bub_params_t *bub_params,
                         mix_params_t *mix_params,
                         array_index_t *array_index,
                         grid_t *grid_size,
                         plane_wave_t *plane_wave)
{
    double2 pos = make_double2(0.0, 0.0);
    host_vector<bubble_t_aos> bub;
    bubble_t_aos init_bubble;
    bubble_t ret_bub;

#ifdef _DEBUG_
    printf("Baking bubbles...\n");
#endif
    for (int i = array_index->istam; i <= array_index->iendm; i++)
    {
        pos.x = ( (double)i - 0.5) * grid_size->dx;
        for (int j = array_index->jstam; j <= array_index->jendm; j++)
        {
            pos.y = ( (double)j - 0.5) * grid_size->dy;

            if (plane_wave->box_size)
            {
                if ((abs(pos.x - plane_wave->fp.x) < 0.5 * plane_wave->box_size)  &&(abs(pos.y - plane_wave->fp.y) < 0.5 * plane_wave->box_size) )
                {
                    init_bubble = bubble_input(	pos,
                                                bub_params->fg0,
                                                *bub_params,
                                                *grid_size,
                                                *plane_wave);
                    bub.push_back(init_bubble);
                }
            }
            else
            {
                init_bubble = bubble_input(	pos,
                                            bub_params->fg0,
                                            *bub_params,
                                            *grid_size,
                                            *plane_wave);
                bub.push_back(init_bubble);
            }
        }
    }
    numBubbles = bub_params->npi = bub.size();

    cudaMallocHost((void**)&ret_bub.ibm, bub.size()*sizeof(int2));
    cudaMallocHost((void**)&ret_bub.ibn, bub.size()*sizeof(int2));
    cudaMallocHost((void**)&ret_bub.pos, bub.size()*sizeof(double2));
    cudaMallocHost((void**)&ret_bub.R_t, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.R_p, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.R_pn, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.R_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.R_nn, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.d1_R_p, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.d1_R_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.PG_p, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.PG_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.PL_p, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.PL_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.PL_m, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.Q_B, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.n_B, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.dt, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.dt_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.re, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.re_n, bub.size()*sizeof(double));
    cudaMallocHost((void**)&ret_bub.v_B, bub.size()*sizeof(double2));
    cudaMallocHost((void**)&ret_bub.v_L, bub.size()*sizeof(double2));


    for (int i = 0; i < bub_params->npi; i++)
    {
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
    printf("%i bubbles initialized.\n\n", bub_params->npi);
#endif
    return ret_bub;
}

// Create a new bubble object based on initial conditions
bubble_t_aos bubble_input(	double2 pos,
                           double fg_in,
                           bub_params_t bub_params,
                           grid_t grid_size,
                           plane_wave_t plane_wave)
{
    bubble_t_aos new_bubble;

    double Pi = acos(-1.0);

    new_bubble.pos 	= pos;

    new_bubble.R_t 	= bub_params.R0;
    new_bubble.R_p	= new_bubble.R_pn	= bub_params.R0;
    new_bubble.R_n	= new_bubble.R_nn	= bub_params.R0;

    new_bubble.d1_R_p	= new_bubble.d1_R_n	= 0.0;

    new_bubble.PG_p	= new_bubble.PG_n	= bub_params.PG0;

    new_bubble.PL_p	= new_bubble.PL_n	= new_bubble.PL_m	= 0.0;

    if (plane_wave.cylindrical)
    {
        new_bubble.n_B	= fg_in * (pos.x * grid_size.dx * grid_size.dy)
                         / (4.0 / 3.0 * Pi * pow(new_bubble.R_t,3));
    }
    else
    {
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
                     const array_index_t array_index)
{
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
    int	jsta1m	=	array_index.jsta1m;
    int	ista2n	=	array_index.ista2n;
    int	jsta2n	=	array_index.jsta2n;
    int	ms	=	array_index.ms;
    int	me	=	array_index.me;
    int	ns	=	array_index.ns;
    int	ne	=	array_index.ne;

    if (PML.X0)
    {
        sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dx * (double)npml);
        if (istam <= npml)
        {
            itmps = max(istam, ms);
            itmpe = min(iendm, npml);
#ifdef _DEBUG_
            printf("Sigma mx :\t");
#endif
            for (int i = itmps; i <= itmpe; i++)
            {
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
    if (PML.X1)
    {
        sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dx * (double)npml);
        if (nx - npml + 1 <= iendm)
        {
            itmps = max(istam, nx - npml + 1);
            itmpe = min(iendm, nx + me);
#ifdef _DEBUG_
            printf("Sigma mx :\t");
#endif
            for (int i = itmps; i <= itmpe; i++)
            {
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
    if (PML.Y0)
    {
        sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dy * (double)npml);
        if (jstam <= npml)
        {
            jtmps = max(jstam, ms);
            jtmpe = min(jendm, npml);
#ifdef _DEBUG_
            printf("Sigma my :\t");
#endif
            for (int j = jtmps; j <= jtmpe; j++)
            {
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
    if (PML.Y1)
    {
        sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dy * (double)npml);
        if (ny - npml + 1 <= jendm)
        {
            jtmps = max(jstam, ny - npml + 1);
            jtmpe = min(jendm, ny + me);
#ifdef _DEBUG_
            printf("Sigma my :\t");
#endif
            for (int j = jtmps; j <= jtmpe; j++)
            {
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
    if (PML.X0)
    {
        sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dx * (double)npml);
        if (istan <= npml - 1)
        {
            itmps = max(istan, ms + ns);
            itmpe = min(iendn, npml - 1);
#ifdef _DEBUG_
            printf("Sigma nx :\t");
#endif
            for (int i = itmps; i <= itmpe; i++)
            {
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
    if (PML.X1)
    {
        sigma_x_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dx * (double)npml);
        if (nx - npml + 1 <= iendn)
        {
            itmps = max(istan, nx - npml + 1);
            itmpe = min(iendn, nx + me + ne + 1);
#ifdef _DEBUG_
            printf("Sigma nx :\t");
#endif
            for (int i = itmps; i <= itmpe; i++)
            {
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
    if (PML.Y0)
    {
        sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dy * (double)npml);
        if (jstan <= npml - 1)
        {
            jtmps = max(jstan, ms + ns);
            jtmpe = min(jendn, npml - 1);
#ifdef _DEBUG_
            printf("Sigma ny :\t");
#endif
            for (int j = jtmps; j <= jtmpe; j++)
            {
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
    if (PML.Y1)
    {
        sigma_y_max = -log(sig) * 0.5 * ((double)order + 1.0)
                      / (dy * (double)npml);
        if (ny - npml + 1 <= jendn)
        {
            jtmps = max(jstan, ny - npml + 1);
            jtmpe = min(jendn, ny + me + ne + 1);
#ifdef _DEBUG_
            printf("Sigma ny :\t");
#endif
            for (int j = jtmps; j <= jtmpe; j++)
            {
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
#ifdef _DEBUG_
    printf("PML generated.\n\n");
#endif
    return sigma;
} // init_sigma()

// Set CUDA runtime flags
void setCUDAflags()
{
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
}

// Allocate and copy variables on device memory
int initialize_CUDA_variables(	grid_t		*grid_size,
                               PML_t		*PML,
                               sim_params_t	*sim_params,
                               plane_wave_t	*plane_wave,
                               array_index_t	*array_index,
                               mix_params_t	*mix_params,
                               bub_params_t	*bub_params)
{

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

    if (bub_params->enabled)
    {
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
    if (bub_params->enabled)
    {
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
    tmp.y = 2.0 * acos(-1.0) / ((double)sim_params->deltaBand) * grid_size->rdx;
    tmp.z = 2.0 * acos(-1.0) / ((double)sim_params->deltaBand) * grid_size->rdy;

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

    if (bub_params->enabled)
    {
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

// Free all CUDA variables
int destroy_CUDA_variables(bub_params_t *bub_params)
{
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

    cutilSafeCall(cudaFreeHost(mixture_h.T));
    cutilSafeCall(cudaFreeHost(mixture_h.vx));
    cutilSafeCall(cudaFreeHost(mixture_h.vy));
    cutilSafeCall(cudaFreeHost(mixture_h.c_sl));
    cutilSafeCall(cudaFreeHost(mixture_h.rho_m));
    cutilSafeCall(cudaFreeHost(mixture_h.rho_l));
    cutilSafeCall(cudaFreeHost(mixture_h.f_g));
    cutilSafeCall(cudaFreeHost(mixture_h.f_gn));
    cutilSafeCall(cudaFreeHost(mixture_h.f_gm));
    cutilSafeCall(cudaFreeHost(mixture_h.k_m));
    cutilSafeCall(cudaFreeHost(mixture_h.C_pm));
    cutilSafeCall(cudaFreeHost(mixture_h.Work));
    cutilSafeCall(cudaFreeHost(mixture_h.Ex));
    cutilSafeCall(cudaFreeHost(mixture_h.Ey));
    cutilSafeCall(cudaFreeHost(mixture_h.p0));
    cutilSafeCall(cudaFreeHost(mixture_h.p));
    cutilSafeCall(cudaFreeHost(mixture_h.pn));

    if (bub_params->enabled)
    {
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

        cutilSafeCall(cudaFreeHost(bubbles_h.ibm));
        cutilSafeCall(cudaFreeHost(bubbles_h.ibn));
        cutilSafeCall(cudaFreeHost(bubbles_h.pos));
        cutilSafeCall(cudaFreeHost(bubbles_h.R_t));
        cutilSafeCall(cudaFreeHost(bubbles_h.R_p));
        cutilSafeCall(cudaFreeHost(bubbles_h.R_pn));
        cutilSafeCall(cudaFreeHost(bubbles_h.R_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.R_nn));
        cutilSafeCall(cudaFreeHost(bubbles_h.d1_R_p));
        cutilSafeCall(cudaFreeHost(bubbles_h.d1_R_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.PG_p));
        cutilSafeCall(cudaFreeHost(bubbles_h.PG_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.PL_p));
        cutilSafeCall(cudaFreeHost(bubbles_h.PL_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.PL_m));
        cutilSafeCall(cudaFreeHost(bubbles_h.Q_B));
        cutilSafeCall(cudaFreeHost(bubbles_h.n_B));
        cutilSafeCall(cudaFreeHost(bubbles_h.dt));
        cutilSafeCall(cudaFreeHost(bubbles_h.dt_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.re));
        cutilSafeCall(cudaFreeHost(bubbles_h.re_n));
        cutilSafeCall(cudaFreeHost(bubbles_h.v_B));
        cutilSafeCall(cudaFreeHost(bubbles_h.v_L));
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

// Checks for any CUDA runtime errors
void checkCUDAError(	const char *msg)
{
#ifdef _DEBUG_
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return;
#else
    return;
#endif
}
