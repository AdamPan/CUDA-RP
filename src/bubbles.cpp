#include "bubbles.h"
using namespace std;

namespace po = boost::program_options;

double fg_max = -1e10, fg_min = 1e10;
double p0_max = -1e10, p0_min = 1e10;
double T_max = -1e10, T_min = 1e10;

string target_dir;

int main (int argc, char *argv[])
{
    // Execute one simulation
    int result = runSimulation(argc, argv);

    return result;
}

int runSimulation(int argc, char *argv[])
{
    time_t start_time = time(NULL);
    bool ok = true;
    string in_file;

    grid_t 	*grid_size;
    PML_t 	*PML;
    sim_params_t	*sim_params;
    bub_params_t	*bub_params;
    plane_wave_t	*plane_wave;
    debug_t		*debug;

	int save_function = none;

    // Load Grid Parameters
    grid_size 	= (grid_t*) malloc(sizeof(grid_t));

//    grid_size->X 	= (int) cf.Value("Grid Dimensions", "X");
//    grid_size->Y 	= (int) cf.Value("Grid Dimensions", "Y");
//    grid_size->LX 	= (double) cf.Value("Grid Dimensions", "LX");
//    grid_size->LY 	= (double) cf.Value("Grid Dimensions", "LY");

    // Load PML Parameters
    PML		= 	(PML_t*) malloc(sizeof(PML_t));

//    PML->X0		= (bool) cf.Value("PML", "X0");
//    PML->X1		= (bool) cf.Value("PML", "X1");
//    PML->Y0		= (bool) cf.Value("PML", "Y0");
//    PML->Y1		= (bool) cf.Value("PML", "Y1");
//    PML->NPML	= (int)	 cf.Value("PML", "Grid Number");
//    PML->order	= (int)  cf.Value("PML", "Order");
//    PML->sigma	= (double) cf.Value("PML", "Sigma");

    // Load simulation parameters
    sim_params	= (sim_params_t*) malloc(sizeof(sim_params_t));

//    sim_params->cfl		= (double) cf.Value("Simulation Parameters", "Courant Number");
//    sim_params->dt0		= (double) cf.Value("Simulation Parameters", "dt0");
//    sim_params->NSTEPMAX	= (int)	cf.Value("Simulation Parameters", "NSTEPMAX");
//    sim_params->TSTEPMAX	= (int)	cf.Value("Simulation Parameters", "TSTEPMAX");
//    sim_params->DATA_SAVE	= (int)	cf.Value("Simulation Parameters", "DATA_SAVE");
//    sim_params->order	= (int)	cf.Value("Simulation Parameters", "Order");
//    sim_params->deltaBand	= (int)	cf.Value("Simulation Parameters", "Smooth Delta Function Band");

    // Load bubble parameters
    bub_params	= (bub_params_t*) calloc(1,sizeof(bub_params_t));

//    bub_params->enabled	= (double) cf.Value("Bubble Properties", "Enable Bubbles");
//    bub_params->fg0	= (double) cf.Value("Bubble Properties", "Initial Void Fraction");
//    bub_params->R0	= (double) cf.Value("Bubble Properties", "Initial Radius");
//    bub_params->PL0	= (double) cf.Value("Bubble Properties", "Initial Pressure");
//    bub_params->T0	= (double) cf.Value("Bubble Properties", "Initial Temperature");
//    bub_params->gam	= (double) cf.Value("Bubble Properties", "Gamma");
//    bub_params->sig	= (double) cf.Value("Bubble Properties", "Surface Tension");
//    bub_params->mu	= (double) cf.Value("Bubble Properties", "Viscosity of Water");
//    bub_params->rho	= (double) cf.Value("Bubble Properties", "Density of Water");
//    bub_params->K0	= (double) cf.Value("Bubble Properties", "K0");

    // Load plane wave settings
    plane_wave	= (plane_wave_t*) calloc(1,sizeof(plane_wave_t));

//    plane_wave->amp		= (double) cf.Value("Plane Wave", "Amplitude");
//    plane_wave->freq	= (double) cf.Value("Plane Wave", "Frequency");
//    plane_wave->f_dist	= (double) cf.Value("Plane Wave", "Focal Distance");
//    plane_wave->box_size	= (double) cf.Value("Plane Wave", "Box Size");
//    plane_wave->cylindrical = (bool)   cf.Value("Plane Wave", "Cylindrical");
//    plane_wave->on_wave	= (int)    cf.Value("Plane Wave", "On Wave");
//    plane_wave->off_wave	= (int)    cf.Value("Plane Wave", "Off Wave");
//    plane_wave->Plane_V	= (bool)   cf.Value("Plane Wave", "Plane Wave (V)");
//    plane_wave->Plane_P	= (bool)   cf.Value("Plane Wave", "Plane Wave (P)");
//    plane_wave->Focused_V	= (bool)   cf.Value("Plane Wave", "Focused Wave (V)");
//    plane_wave->Focused_P	= (bool)   cf.Value("Plane Wave", "Focused Wave (P)");

    // Load Debug Parameters
    debug		= (debug_t*) malloc(sizeof(debug_t));
//    debug->display	= (int)	cf.Value("Debug", "Display Lines", 0);
//    debug->fg	= (bool)cf.Value("Debug", "Show fg", 0);
//    debug->p0	= (bool)cf.Value("Debug", "Show p", 0);
//    debug->T	= (bool)cf.Value("Debug", "Show T", 0);
//    debug->vxy	= (bool)cf.Value("Debug", "Show v", 0);
//    debug->bubbles	= (bool)cf.Value("Debug", "Show Bubbles", 0);
//    if (bub_params->enabled == 0)
//    {
//        debug->bubbles = 0;
//    }


	po::options_description cmd_opts("Command line options"),
							generic_opts("Generic options"),
							file_opts("Configuration file options"),
							hidden_opts("Hidden options");
	cmd_opts.add_options()
		("help,h", "Output this help message.")
		("input-file", po::value<string>()->multitoken(), "input file")
	;
	generic_opts.add_options()
		("ascii,a", "Save plaintext files.")
		("sph,s", "Save sph files. This option has precedence over --ascii.")
		("directory,d", po::value<string>()->default_value("./rawdata/"), "Save into a target directory (default: ./rawdata/).")
	;
	file_opts.add_options()
		("grid.x", po::value<int>(&grid_size->X))
		("grid.y", po::value<int>(&grid_size->Y))
		("grid.lx", po::value<double>(&grid_size->LX))
		("grid.ly", po::value<double>(&grid_size->LY))
		("pml.x0", po::value<bool>(&PML->X0))
		("pml.x1", po::value<bool>(&PML->X1))
		("pml.y0", po::value<bool>(&PML->Y0))
		("pml.y1", po::value<bool>(&PML->Y1))
		("pml.grid_number", po::value<int>(&PML->NPML))
		("pml.sigma", po::value<double>(&PML->sigma))
		("pml.order", po::value<int>(&PML->order))
		("simulation.cfl", po::value<double>(&sim_params->cfl))
		("simulation.dt0", po::value<double>(&sim_params->dt0))
		("simulation.n_step_max", po::value<int>(&sim_params->NSTEPMAX))
		("simulation.t_step_max", po::value<double>(&sim_params->TSTEPMAX))
		("simulation.save_interval", po::value<int>(&sim_params->DATA_SAVE))
		("simulation.order", po::value<int>(&sim_params->order))
		("simulation.smooth_delta_band", po::value<int>(&sim_params->deltaBand))
		("plane_wave.amp", po::value<double>(&plane_wave->amp))
		("plane_wave.freq", po::value<double>(&plane_wave->freq))
		("plane_wave.focal_dist", po::value<double>(&plane_wave->f_dist))
		("plane_wave.box_size", po::value<double>(&plane_wave->box_size))
		("plane_wave.cylindrical", po::value<bool>(&plane_wave->cylindrical))
		("plane_wave.on_wave", po::value<int>(&plane_wave->on_wave))
		("plane_wave.off_wave", po::value<int>(&plane_wave->off_wave))
		("plane_wave.plane_wave_v", po::value<bool>(&plane_wave->Plane_V))
		("plane_wave.plane_wave_p", po::value<bool>(&plane_wave->Plane_P))
		("plane_wave.focused_wave_v", po::value<bool>(&plane_wave->Focused_V))
		("plane_wave.focused_wave_p", po::value<bool>(&plane_wave->Focused_P))
		("bubbles.enable", po::value<bool>(&bub_params->enabled)->default_value(false)->implicit_value(true))
		("bubbles.fg0", po::value<double>(&bub_params->fg0))
		("bubbles.R0", po::value<double>(&bub_params->R0))
		("bubbles.P0", po::value<double>(&bub_params->PL0))
		("bubbles.T0", po::value<double>(&bub_params->T0))
		("bubbles.gamma", po::value<double>(&bub_params->gam))
		("bubbles.liquid_surface_tension", po::value<double>(&bub_params->sig))
		("bubbles.liquid_viscosity", po::value<double>(&bub_params->mu))
		("bubbles.liquid_density", po::value<double>(&bub_params->rho))
		("bubbles.K0", po::value<double>(&bub_params->K0))
		("debug.display_dim", po::value<int>(&debug->display)->default_value(4))
		("debug.fg", po::value<bool>(&debug->fg  )->default_value(true )->implicit_value(true))
		("debug.T",  po::value<bool>(&debug->T  )->default_value(true )->implicit_value(true))
		("debug.p",  po::value<bool>(&debug->p0 )->default_value(false)->implicit_value(true))
		("debug.v",  po::value<bool>(&debug->vxy)->default_value(false)->implicit_value(true))
		("debug.bubbles", po::value<bool>(&debug->bubbles)->default_value(10))
	;
	hidden_opts.add_options()
		("input-file", po::value< vector<string> >(), "input file")
	;
	
	po::options_description cmd_line_opts;
	cmd_line_opts.add(cmd_opts).add(generic_opts).add(hidden_opts);
	po::options_description cfg_file_opts;
	cfg_file_opts.add(file_opts).add(generic_opts).add(hidden_opts);
	po::options_description visible_opts("Options");
	visible_opts.add(cmd_opts).add(generic_opts);

	po::positional_options_description pos_opts;
	pos_opts.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(cmd_line_opts).positional(pos_opts).allow_unregistered().run(), vm);
	po::notify(vm);
	
	if (vm.count("help"))
	{
		cout << visible_opts << "\n";
		return 1;
	}
	
	if (vm.count("ascii"))
	{
		save_function |= ascii;
	}
	
	if (vm.count("sph"))
	{
		save_function |= sph;
	}

	if (vm.count("input-file"))
	{
		in_file = vm["input-file"].as<string>();
	}

	if (vm.count("directory"))
	{
		target_dir = vm["directory"].as<string>();
	}

    // Read Configuration File
    // Compile a full list and define a configuration schema.

	ifstream ifs(in_file.c_str());
	po::store(po::parse_config_file(ifs, cfg_file_opts, true), vm);
	po::notify(vm);

	cout << "Input file is " << in_file << endl;
	cout << "Saving results to " << target_dir << endl;
	cout << "Save format is " << (vm.count("sph") ? "sph" : "ASCII plaintext") << endl;

    array_index_t *array_index = (array_index_t *) calloc(1, sizeof(array_index_t));

    // Enter main simulation loop
    if (ok && solve_bubbles(array_index, grid_size, PML, sim_params, bub_params, plane_wave, debug, save_function))
    {
        exit(EXIT_FAILURE);
    }

    // tell us the time

    cout << "The program took " << time(NULL) - start_time << " seconds to run" << endl;
    ofstream runtime;
    runtime.open((target_dir + "../runtime.txt").c_str());
    runtime << "Finished in " << time(NULL) - start_time << " seconds" << endl;
    runtime.close();

    ofstream file;
//    if (debug->T)
//    {
//        T_min  = (double)cf.Value("Debug", "Min T",  T_min);
//        T_max  = (double)cf.Value("Debug", "Max T",  T_max);
//        file.open((target_dir + "T_minmax.txt").c_str());
//        file << T_min << endl << T_max << endl;
//        file.close();
//    }
//    if (debug->p0)
//    {
//        p0_min = (double)cf.Value("Debug", "Min p",  p0_min);
//        p0_max = (double)cf.Value("Debug", "Max p",  p0_max);
//        file.open((target_dir + "p0_minmax.txt").c_str());
//        file << p0_min << endl << p0_max << endl;
//        file.close();
//    }
//    if (debug->fg)
//    {
//        fg_min = (double)cf.Value("Debug", "Min fg", fg_min);
//        fg_max = (double)cf.Value("Debug", "Max fg", fg_max);
//        file.open((target_dir + "fg_minmax.txt").c_str());
//        file << fg_min << endl << fg_max << endl;
//        file.close();
//    }
    return 0;
}
