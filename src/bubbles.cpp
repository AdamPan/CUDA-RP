
#include "bubbles.h"
#include "bubble_CUDA.h"


using namespace std;


extern int numBubbles;
extern int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;
double fg_max = -1e10, fg_min = 1e10;
double p0_max = -1e10, p0_min = 1e10;
double T_max = -1e10, T_min = 1e10;
string target_folder = "./rawdata/";

// Forward declarations
int runSimulation(int argc, char **argv);

int main (int argc, char **argv)
{
    // Execute
    int result = runSimulation(argc, argv);

    return result;
}

extern "C" int initialize_folders()
{
    ofstream out_file;
    stringstream out;

    string out_dir = target_folder;
    string clear_dir = "rm -rf " + target_folder;
    string make_dir = "mkdir " + target_folder;

    if (system(clear_dir.c_str()))
    {
        exit(EXIT_FAILURE);
    }
    if (system(make_dir.c_str()))
    {
        exit(EXIT_FAILURE);
    }
    return 0;
}

extern "C" void *save_step(void *threadArg)
{
    struct output_plan_t plan;
    plan = *((struct output_plan_t*) threadArg);
    mixture_t mixture_h = plan.mixture_h;
    bubble_t bubbles_h = plan.bubbles_h;
    int step = plan.step;
    array_index_t *array_index = plan.array_index;
    grid_t *grid_size = plan.grid_size;
    sim_params_t *sim_params = plan.sim_params;
    plane_wave_t *plane_wave = plan.plane_wave;
    debug_t *debug = plan.debug;

    int index = 0;

    ofstream out_file;
    stringstream out;
    string out_dir = target_folder;
    if (debug->fg)
    {
        out.str("");
        out << out_dir << "fg_step_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = (plane_wave->cylindrical ? -grid_size->X : 0); i <= grid_size->X; i++)
            {
                index = i2m * (j - array_index->jsta2m) + abs(i) - array_index->ista2m;
                out_file << (double)i*grid_size->dx << "\t" << (double)j*grid_size->dy << "\t" << mixture_h.f_g[index] << endl;
                fg_max = max(fg_max, mixture_h.f_g[index]);
                fg_min = min(fg_min, mixture_h.f_g[index]);
            }
            out_file << endl;
        }
        out_file.close();
    }

    if (debug->p0)
    {
        out.str("");
        out << out_dir << "p0_step_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = (plane_wave->cylindrical ? -grid_size->X : 0); i <= grid_size->X; i++)
            {
                index = i1m * (j - array_index->jsta1m) + abs(i) - array_index->ista1m;
                out_file << (double)i*grid_size->dx << "\t" << (double)j*grid_size->dy << "\t" << mixture_h.p0[index] << endl;
                p0_max = max(p0_max, mixture_h.p0[index]);
                p0_min = min(p0_min, mixture_h.p0[index]);
            }
            out_file << endl;
        }
        out_file.close();
    }

    if (debug->T)
    {
        out.str("");
        out << out_dir << "T_step_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = (plane_wave->cylindrical ? -grid_size->X : 0); i <= grid_size->X; i++)
            {
                index = i2m * (j - array_index->jsta2m) + abs(i) - array_index->ista2m;
                out_file << (double)i*grid_size->dx << "\t" << (double)j*grid_size->dy << "\t" << mixture_h.T[index] << endl;
                T_max = max(T_max, mixture_h.T[index]);
                T_min = min(T_min, mixture_h.T[index]);
            }
            out_file << endl;
        }
        out_file.close();
    }

    if (debug->vxy)
    {
        out.str("");
        out << out_dir << "vx_step_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = (plane_wave->cylindrical ? -grid_size->X : 0); i <= grid_size->X; i++)
            {
                out_file << (double)i*grid_size->dx << "\t" << (double)j*grid_size->dy << "\t" << mixture_h.vx[i2n * (j - array_index->jsta2m) + abs(i) - array_index->ista2n] << endl;
            }
            out_file << endl;
        }
        out_file.close();

        out.str("");
        out << out_dir << "vy_step_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = (plane_wave->cylindrical ? -grid_size->X : 0); i <= grid_size->X; i++)
            {
                out_file << (double)i*grid_size->dx << "\t" << (double)j*grid_size->dy << "\t" << mixture_h.vy[i2m * (j - array_index->jsta2n) + abs(i) - array_index->ista2m] << endl;
            }
            out_file << endl;
        }
        out_file.close();
    }
    if (debug->bubbles)
    {
        out.str("");
        out << out_dir << "bubble_radii_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int i = 0; i < numBubbles; i++)
        {
            out_file << bubbles_h.pos[i].x << "\t" << bubbles_h.pos[i].y << "\t" << bubbles_h.R_t[i] <<  endl;
        }
//			if (plane_wave->cylindrical){
//				for (int i = 0; i < numBubbles; i++){
//					out_file << -bubbles_h.pos[i].x << "\t" << bubbles_h.pos[i].y << "\t" << bubbles_h.R_t[i] <<  endl;
//				}
//			}
        out_file.close();

        out.str("");
        out << out_dir << "bubble_PG_";
        out.width(5);
        out.fill('0');
        out << step << ".txt";
        out_file.open((out.str()).c_str());

        for (int i = 0; i < numBubbles; i++)
        {
            out_file << bubbles_h.pos[i].x << "\t" << bubbles_h.pos[i].y << "\t" << bubbles_h.PG_p[i] << endl;
        }
//			if (plane_wave->cylindrical){
//				for (int i = 0; i < numBubbles; i++){
//					out_file << -bubbles_h.pos[i].x << "\t" << bubbles_h.pos[i].y << "\t" << bubbles_h.PG_p[i] << endl;
//				}
//			}
        out_file.close();
    }
    return 0;
}


int runSimulation(int argc, char **argv)
{
    time_t start_time = time(NULL);
    bool ok = true;
    string in_file, out_file;

    grid_t 	*grid_size;
    PML_t 	*PML;
    sim_params_t	*sim_params;
    bub_params_t	*bub_params;
    plane_wave_t	*plane_wave;
    debug_t		*debug;

    if (argc > 1)
    {
        in_file		=	argv[1];
    }
    else
    {
        cout << "Invalid number of arguments, please provide an input and output file" << endl;
        ok = false;
    }
    if (argc > 2)
    {
        target_folder     =   argv[2];
    }

    // Read Data
    // Needs to actually read data we need. Compile a full list and define a configuration schema.
    if (ok)
    {
        ConfigFile cf(in_file);

        // Load Grid Parameters
        grid_size 	= (grid_t*) malloc(sizeof(grid_t));

        grid_size->X 	= (int) cf.Value("Grid Dimensions", "X");
        grid_size->Y 	= (int) cf.Value("Grid Dimensions", "Y");
        grid_size->LX 	= (double) cf.Value("Grid Dimensions", "LX");
        grid_size->LY 	= (double) cf.Value("Grid Dimensions", "LY");

        // Load PML Parameters
        PML		= 	(PML_t*) malloc(sizeof(PML_t));

        PML->X0		= (bool) cf.Value("PML", "X0");
        PML->X1		= (bool) cf.Value("PML", "X1");
        PML->Y0		= (bool) cf.Value("PML", "Y0");
        PML->Y1		= (bool) cf.Value("PML", "Y1");
        PML->NPML	= (int)	 cf.Value("PML", "Grid Number");
        PML->order	= (int)  cf.Value("PML", "Order");
        PML->sigma	= (double) cf.Value("PML", "Sigma");

        // Load simulation parameters
        sim_params	= (sim_params_t*) malloc(sizeof(sim_params_t));

        sim_params->cfl		= (double) cf.Value("Simulation Parameters", "Courant Number");
        sim_params->dt0		= (double) cf.Value("Simulation Parameters", "dt0");
        sim_params->NSTEPMAX	= (int)	cf.Value("Simulation Parameters", "NSTEPMAX");
        sim_params->TSTEPMAX	= (int)	cf.Value("Simulation Parameters", "TSTEPMAX");
        sim_params->DATA_SAVE	= (int)	cf.Value("Simulation Parameters", "DATA_SAVE");
        sim_params->order	= (int)	cf.Value("Simulation Parameters", "Order");
        sim_params->deltaBand	= (int)	cf.Value("Simulation Parameters", "Smooth Delta Function Band");

        // Load bubble parameters
        bub_params	= (bub_params_t*) calloc(1,sizeof(bub_params_t));

        bub_params->enabled	= (double) cf.Value("Bubble Properties", "Enable Bubbles");
        bub_params->fg0	= (double) cf.Value("Bubble Properties", "Initial Void Fraction");
        bub_params->R0	= (double) cf.Value("Bubble Properties", "Initial Radius");
        bub_params->PL0	= (double) cf.Value("Bubble Properties", "Initial Pressure");
        bub_params->T0	= (double) cf.Value("Bubble Properties", "Initial Temperature");
        bub_params->gam	= (double) cf.Value("Bubble Properties", "Gamma");
        bub_params->sig	= (double) cf.Value("Bubble Properties", "Surface Tension");
        bub_params->mu	= (double) cf.Value("Bubble Properties", "Viscosity of Water");
        bub_params->rho	= (double) cf.Value("Bubble Properties", "Density of Water");
        bub_params->K0	= (double) cf.Value("Bubble Properties", "K0");

        // Load plane wave settings
        plane_wave	= (plane_wave_t*) calloc(1,sizeof(plane_wave_t));

        plane_wave->amp		= (double) cf.Value("Plane Wave", "Amplitude");
        plane_wave->freq	= (double) cf.Value("Plane Wave", "Frequency");
        plane_wave->f_dist	= (double) cf.Value("Plane Wave", "Focal Distance");
        plane_wave->box_size	= (double) cf.Value("Plane Wave", "Box Size");
        plane_wave->cylindrical = (bool)   cf.Value("Plane Wave", "Cylindrical");
        plane_wave->on_wave	= (int)    cf.Value("Plane Wave", "On Wave");
        plane_wave->off_wave	= (int)    cf.Value("Plane Wave", "Off Wave");
        plane_wave->Plane_V	= (bool)   cf.Value("Plane Wave", "Plane Wave (V)");
        plane_wave->Plane_P	= (bool)   cf.Value("Plane Wave", "Plane Wave (P)");
        plane_wave->Focused_V	= (bool)   cf.Value("Plane Wave", "Focused Wave (V)");
        plane_wave->Focused_P	= (bool)   cf.Value("Plane Wave", "Focused Wave (P)");

        // Load Debug Parameters
        debug		= (debug_t*) malloc(sizeof(debug_t));
        debug->display	= (int)	cf.Value("Debug", "Display Lines");
        debug->fg	= (bool)cf.Value("Debug", "Show fg");
        debug->p0	= (bool)cf.Value("Debug", "Show p");
        debug->T	= (bool)cf.Value("Debug", "Show T");
        debug->vxy	= (bool)cf.Value("Debug", "Show v");
        debug->bubbles	= (bool)cf.Value("Debug", "Show Bubbles");
        if (bub_params->enabled == 0)
        {
            debug->bubbles = 0;
        }
    }

    array_index_t *array_index = (array_index_t *) calloc(1, sizeof(array_index_t));

    // Enter main simulation loop
    if (ok && solve_bubbles(array_index, grid_size, PML, sim_params, bub_params, plane_wave, debug, argc, argv))
    {
        exit(EXIT_FAILURE);
    }

    // tell us the time

    cout << "The program took " << time(NULL) - start_time << " seconds to run" << endl;
    ofstream runtime;
    runtime.open("runtime.txt");
    runtime << "Finished in " << time(NULL) - start_time << " seconds" << endl;
    runtime.close();
    
    ofstream file;
    if (debug->T){
        file.open((target_folder + "T_minmax.txt").c_str());
        file << T_min << endl << T_max << endl;
        file.close();
    }
    if (debug->p0){
        file.open((target_folder + "p0_minmax.txt").c_str());
        file << p0_min << endl << p0_max << endl;
        file.close();
    }
    if (debug->fg){
        file.open((target_folder + "fg_minmax.txt").c_str());
        file << fg_min << endl << fg_max << endl;
        file.close();
    }
    return 0;
}
