#include "bubbles.h"
using namespace std;

namespace po = boost::program_options;

extern thrust::host_vector<double2> focalpoint;
extern thrust::host_vector<double2> control;

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

    // Grid parameters
    grid_size 	= (grid_t*) malloc(sizeof(grid_t));

    // PML parameters
    PML		= 	(PML_t*) malloc(sizeof(PML_t));

    // Simulation parameters
    sim_params	= (sim_params_t*) malloc(sizeof(sim_params_t));

    // Bubble parameters
    bub_params	= (bub_params_t*) calloc(1,sizeof(bub_params_t));

    // Plane wave settings
    plane_wave	= (plane_wave_t*) calloc(1,sizeof(plane_wave_t));

    // Debug parameters
    debug		= (debug_t*) malloc(sizeof(debug_t));

	po::options_description cmd_opts("Command line options"),
							generic_opts("Generic options"),
							file_opts("Configuration file options"),
							hidden_opts("Hidden options");
	cmd_opts.add_options()
		("help,h", "Output this help message.")
		("help-all", "Output a complete list of options.")
	;
	generic_opts.add_options()
		("ascii,a", "Save plaintext files.")
		("sph,s", "Save sph files. This option has\nprecedence over --ascii.")
		("directory,d", po::value<string>()->default_value("./rawdata/"), "Save into a target directory.")
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
		("plane_wave.pid", po::value<bool>(&plane_wave->pid)->default_value(false))
		("plane_wave.pid_start_step", po::value<int>(&plane_wave->pid_start_step)->default_value(1))
		("plane_wave.init_control", po::value<double>(&plane_wave->init_control)->default_value(0.0))
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
		("debug.fg_min", po::value<double>())
		("debug.fg_max", po::value<double>())
		("debug.T",  po::value<bool>(&debug->T  )->default_value(true )->implicit_value(true))
		("debug.T_min", po::value<double>())
		("debug.T_max", po::value<double>())
		("debug.p",  po::value<bool>(&debug->p0 )->default_value(false)->implicit_value(true))
		("debug p_min", po::value<double>())
		("debug.p_max", po::value<double>())
		("debug.v",  po::value<bool>(&debug->vxy)->default_value(false)->implicit_value(true))
		("debug.v_min", po::value<double>())
		("debug.v_max", po::value<double>())
		("debug.bubbles", po::value<bool>(&debug->bubbles)->default_value(10))
	;
	hidden_opts.add_options()
		("input-file", po::value<string>(), "input file")
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
	
	if (vm.count("help-all")){
		visible_opts.add(file_opts);
		cout << visible_opts << endl;
		return 0;
	}
	
	if (vm.count("help"))
	{
		cout << visible_opts << endl;
		return 0;
	}

	if (vm.count("input-file"))
	{
		in_file = vm["input-file"].as<string>();
	}
	else
	{
		cout << "Please specify an input configuration file" << endl;
		return 1;
	}

	if (vm.count("directory"))
	{
		target_dir = vm["directory"].as<string>();
		if (target_dir.find_last_of("/") != target_dir.length() - 1)
		{
			target_dir += "/";
		}
	}

    // Read Configuration File
    // Compile a full list and define a configuration schema.

	ifstream ifs(in_file.c_str());
	po::store(po::parse_config_file(ifs, cfg_file_opts, true), vm);
	po::notify(vm);

	if (vm.count("ascii"))
	{
		save_function |= ascii;
	}
	
	if (vm.count("sph"))
	{
		save_function |= sph;
	}

	cout << "Input file is " << in_file << endl;
	cout << "Saving results to " << target_dir << endl;
	cout << "Save format is " << (save_function == sph ? "sph" : "ASCII plaintext") << endl << endl;

    array_index_t *array_index = (array_index_t *) calloc(1, sizeof(array_index_t));

    // Enter main simulation loop
    if (ok && solve_bubbles(array_index, grid_size, PML, sim_params, bub_params, plane_wave, debug, save_function))
    {
        exit(EXIT_FAILURE);
    }

    // tell us the time

    cout << "The program took " << time(NULL) - start_time << " seconds to run" << endl;
    ofstream runtime;
    runtime.open((target_dir + "runtime.txt").c_str());
    runtime << "Finished in " << time(NULL) - start_time << " seconds" << endl;
    runtime.close();

	ofstream fp_file;
	fp_file.open((target_dir + "focalpoint_actual_T.txt").c_str());
	for (int i = 0; i < focalpoint.size(); i++)
	{
		fp_file << i << "\t" << focalpoint[i].y << endl;
	}
	fp_file.close();

	fp_file.open((target_dir + "focalpoint_controller_T.txt").c_str());
	for (int i = 0; i < control.size(); i++)
	{
		fp_file << i + plane_wave->pid_start_step << "\t" << control[i].y << endl;
	}
	fp_file.close();


    ofstream file;
    if (debug->T)
    {
		if (vm.count("debug.T_min")) T_min = vm["debug.T_min"].as<double>();
        if (vm.count("debug.T_max")) T_max = vm["debug.T_max"].as<double>();
        file.open((target_dir + "T_minmax.txt").c_str());
        file << T_min << endl << T_max << endl;
        file.close();
    }
    if (debug->p0)
    {
        if (vm.count("debug.p_min")) p0_min = vm["debug.p_min"].as<double>();
        if (vm.count("debug.p_max")) p0_max = vm["debug.p_max"].as<double>();
        file.open((target_dir + "p0_minmax.txt").c_str());
        file << p0_min << endl << p0_max << endl;
        file.close();
    }
    if (debug->fg)
    {
        if (vm.count("debug.fg_min")) fg_min = vm["debug.fg_min"].as<double>();
        if (vm.count("debug.fg_max")) fg_max = vm["debug.fg_max"].as<double>();
        file.open((target_dir + "fg_minmax.txt").c_str());
        file << fg_min << endl << fg_max << endl;
        file.close();
    }
    return 0;
}
