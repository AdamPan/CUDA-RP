
#include "bubbles.h"
#include "bubble_CUDA.h"


using namespace std;


extern int numBubbles;
extern int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;

// Forward declarations
int runSimulation(int argc, char **argv);
int send_to_file(thrust::host_vector<solution_space> solution, sim_params_t *sim_params, debug_t *debug);

int main (int argc, char **argv)
{
	// Execute
	int result = runSimulation(argc, argv);
	
	return (int)result;
}

int send_to_file(thrust::host_vector<solution_space> solution, sim_params_t *sim_params, debug_t *debug){
	ofstream out_file;
	stringstream out;
	string out_dir = "./results/";
	string clear_dir = "rm -rf ./results/*";
	if(system(clear_dir.c_str())){exit(EXIT_FAILURE);}
	if(system("clear")){exit(EXIT_FAILURE);}
	
	cout << "Simulation complete!" << endl << endl;
	cout << "Saving results to folder: " << out_dir << endl << endl;
	
	if (debug->fg){
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "fg_step_";
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for (int j = 0; j < j2m; j++){
				for (int i = 0; i < i2m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].f_g[i2m * j + i] << endl;
				}
				out_file << endl;
			}
			out_file.close();
			cout << "\r" << "Saving FG : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
	}

	if (debug->p0){
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "p0_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for (int j = 0; j < j1m; j++){
				for (int i = 0; i < i1m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].p0[i1m * j + i] << endl;
				}
				out_file << endl;				
			}
			out_file.close();
			cout << "\r" << "Saving P0 : " << (int) 100 * (k + 1) / solution.size() << "% done";
			cout.flush();
		}
		cout << endl;
	}
	
	if (debug->pxy){
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "px_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for(int j = 0; j < j1m; j++){
				for (int i = 0; i < i1m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].p[i1m * j + i].x << endl;
				}
				out_file << endl;				
			}
			out_file.close();
			cout << "\r" << "Saving PX : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "py_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for(int j = 0; j < j1m; j++){
				for (int i = 0; i < i1m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].p[i1m * j + i].y << endl;
				}
				out_file << endl;
			}
			out_file.close();
			cout << "\r" << "Saving PY : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
	}
	
	if (debug->T){
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "T_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for(int j = 0; j < j2m; j++){
				for (int i = 0; i < i2m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].T[i2m * j + i] << endl;
				}
				out_file << endl;
			}
			out_file.close();
			cout << "\r" << "Saving T : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
	}
	
	if (debug->vxy){
		for (int k = 0; k < solution.size(); k ++){
			out.str("");
			out << out_dir << "vx_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for (int j = 0; j < j2m; j++){
				for (int i = 0; i < i2n; i++){
					out_file << i << "\t" << j << "\t" << solution[k].vx[i2n * j + i] << endl;
				}
				out_file << endl;
			}
			out_file.close();
			cout << "\r" << "Saving VX : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
		for (int k = 0; k < solution.size(); k ++){
			out.str("");
			out << out_dir << "vy_step_" ;
			out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
			
			for (int j = 0; j < j2n; j++){
				for (int i = 0; i < i2m; i++){
					out_file << i << "\t" << j << "\t" << solution[k].vy[i2m * j + i] << endl;
				}
				out_file << endl;
			}
			out_file.close();
			cout << "\r" << "Saving VY : " << (int) 100 * (k + 1) / solution.size() << "% done" ;
			cout.flush();
		}
		cout << endl;
	}
	if (debug->bubbles){
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "bubble_radii_" ;
				out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
		
			for (int i = 0; i < numBubbles; i++){
				out_file << solution[k].pos[i].x << "\t" << solution[k].pos[i].y << "\t" << solution[k].R_t[i] <<  endl;
			}
			out_file.close();
			cout << "\r" << "Saving Bubble Radii : " << (int) 100 * (k + 1) / solution.size() << "% done" ;		
		}
		for (int k = 0; k < solution.size(); k++){
			out.str("");
			out << out_dir << "bubble_PL_" ;
				out.width(5); out.fill('0'); out << (k + 1) * sim_params->DATA_SAVE << ".txt";
			out_file.open((out.str()).c_str());
		
			for (int i = 0; i < numBubbles; i++){
				out_file << solution[k].pos[i].x << "\t" << solution[k].pos[i].y << "\t" << solution[k].PL_p[i] << endl;
			}
			out_file.close();
			cout << "\r" << "Saving Bubble Liquid Pressure : " << (int) 100 * (k + 1) / solution.size() << "% done" ;		
		}
	}
	return 0;
	
}


int runSimulation(int argc, char **argv)
{
	time_t start_time = time(NULL);
	bool ok = true;
	string in_file, out_file;
	
	thrust::host_vector<solution_space> solution;
	
	grid_t 	*grid_size;
	PML_t 	*PML;
	sim_params_t	*sim_params;
	bub_params_t	*bub_params;
	plane_wave_t	*plane_wave;
	debug_t		*debug;
	
	if (argc == 2){
		in_file		=	argv[1];
	}
	else{
		cout << "Invalid number of arguments, please provide an input and output file" << endl;
		ok = false;
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
		sim_params->WRITE_20	= (int)	cf.Value("Simulation Parameters", "WRITE_20");
		sim_params->order	= (int)	cf.Value("Simulation Parameters", "Order");
		sim_params->deltaBand	= (int)	cf.Value("Simulation Parameters", "Smooth Delta Function Band");
		
		// Load bubble parameters
		bub_params	= (bub_params_t*) calloc(1,sizeof(bub_params_t));
		
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
		
		#ifdef _DEBUG_
		// Load Debug Parameters
		debug		= (debug_t*) malloc(sizeof(debug_t));
		debug->display	= (int)	cf.Value("Debug", "Display Lines");
		debug->fg	= (bool)cf.Value("Debug", "Show fg");
		debug->p0	= (bool)cf.Value("Debug", "Show p");
		debug->pxy	= (bool)cf.Value("Debug", "Show p components");
		debug->T	= (bool)cf.Value("Debug", "Show T");
		debug->vxy	= (bool)cf.Value("Debug", "Show v");
		debug->bubbles	= (bool)cf.Value("Debug", "Show Bubbles");
		#endif
	}
	
	// Enter main simulation loop
	if (ok)
	{
		solution = solve_bubbles(grid_size, PML, sim_params, bub_params, plane_wave, debug, argc, argv);
	}
	
	// tell us the time
	
	cout << "The program took " << time(NULL) - start_time << " seconds to run" << endl;
	ofstream runtime;
	runtime.open("runtime.txt");
	runtime << "Finished in " << time(NULL) - start_time << " seconds" << endl;
	runtime.close();
	// Output the solution
	//send_to_file(solution, sim_params, debug);
	
	return 0;
}
