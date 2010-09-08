#include <output_styles.h>

using namespace std;

extern int numBubbles;
extern int j0m, j0n, i1m, j1m, i1n, j1n, i2m, j2m, i2n, j2n, m1Vol, m2Vol, v_xVol, v_yVol, E_xVol, E_yVol;
extern double fg_max, fg_min;
extern double p0_max, p0_min;
extern double T_max, T_min;
extern string target_dir;



extern "C" int initialize_folders()
{
    ofstream out_file;
    stringstream out;

    string out_dir = target_dir;
    string clear_dir = "rm -rf " + target_dir;
    string make_dir = "mkdir " + target_dir;

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



extern "C" void *save_sph(void *threadArg)
{
    struct output_plan_t plan;
    plan = *((struct output_plan_t*) threadArg);
    mixture_t mixture_h = plan.mixture_h;
    bubble_t bubbles_h = plan.bubbles_h;

    int step = plan.step;
    double tstep = plan.tstep;
    array_index_t *array_index = plan.array_index;
    grid_t *grid_size = plan.grid_size;
    sim_params_t *sim_params = plan.sim_params;
    plane_wave_t *plane_wave = plan.plane_wave;
    debug_t *debug = plan.debug;

    int index = 0;
    int dim[3] = { grid_size->X + 1, grid_size->Y + 1, 1 };
    float origin[3] = { 0.0f, 0.0f, 0.0f };
    float pitch[3] = { (float)grid_size->dx, (float)grid_size->dy, 1.0f };

    SphData fg_sph, p0_sph, T_sph;
    stringstream out;
    string out_dir = target_dir;

    if (debug->fg)
    {
        fg_sph.Init(dim, origin, pitch, tstep, step);
        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = 0; i <= grid_size->X; i++)
            {
                index = i2m * (j - array_index->jsta2m) + abs(i) - array_index->ista2m;
                fg_sph.SetValue(i, j, 0, (float)mixture_h.f_g[index]);
                fg_max = max(fg_max, mixture_h.f_g[index]);
                fg_min = min(fg_min, mixture_h.f_g[index]);
            }
        }
    }

    if (debug->p0)
    {
        p0_sph.Init(dim, origin, pitch, tstep, step);
        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = 0; i <= grid_size->X; i++)
            {
                index = i1m * (j - array_index->jsta1m) + abs(i) - array_index->ista1m;
                p0_sph.SetValue(i, j, 0, (float)mixture_h.p0[index]);
                p0_max = max(p0_max, mixture_h.p0[index]);
                p0_min = min(p0_min, mixture_h.p0[index]);
            }
        }
    }

    if (debug->T)
    {
        T_sph.Init(dim, origin, pitch, tstep, step);
        for (int j = 0; j <= grid_size->Y; j++)
        {
            for (int i = 0; i <= grid_size->X; i++)
            {
                index = i2m * (j - array_index->jsta2m) + abs(i) - array_index->ista2m;
                T_sph.SetValue(i, j, 0, (float)mixture_h.T[index]);
                T_max = max(T_max, mixture_h.T[index]);
                T_min = min(T_min, mixture_h.T[index]);
            }
        }
    }

    if (debug->fg)
    {
        out.str("");
        out << out_dir << "fg_step_";
        out.width(5);
        out.fill('0');
        out << step << ".sph";
        fg_sph.SaveSph(out.str());
        fg_sph.Deallocate();
    }

    if (debug->p0)
    {
        out.str("");
        out << out_dir << "p0_step_";
        out.width(5);
        out.fill('0');
        out << step << ".sph";
        p0_sph.SaveSph(out.str());
        p0_sph.Deallocate();
    }

    if (debug->T)
    {
        out.str("");
        out << out_dir << "T_step_";
        out.width(5);
        out.fill('0');
        out << step << ".sph";
        T_sph.SaveSph(out.str());
        T_sph.Deallocate();
    }
}

extern "C" void *save_ascii(void *threadArg)
{
    struct output_plan_t plan;
    plan = *((struct output_plan_t*) threadArg);
    mixture_t mixture_h;
    bubble_t bubbles_h;

    int step = plan.step;
    array_index_t *array_index = plan.array_index;
    grid_t *grid_size = plan.grid_size;
    sim_params_t *sim_params = plan.sim_params;
    plane_wave_t *plane_wave = plan.plane_wave;
    debug_t *debug = plan.debug;

    int index = 0;

    if (debug->T)  mixture_h.T   = (double*) calloc(m2Vol,  sizeof(double));
    if (debug->p0) mixture_h.p0  = (double*) calloc(m1Vol,  sizeof(double));
    if (debug->fg) mixture_h.f_g = (double*) calloc(m2Vol,  sizeof(double));
    if (debug->vxy)mixture_h.vx  = (double*) calloc(v_xVol, sizeof(double));
    if (debug->vxy)mixture_h.vy  = (double*) calloc(v_yVol, sizeof(double));

    if (debug->bubbles)
    {
        bubbles_h.pos  = (double2*)calloc(numBubbles, sizeof(double2));
        bubbles_h.R_t  = (double*) calloc(numBubbles, sizeof(double));
        bubbles_h.PG_p = (double*) calloc(numBubbles, sizeof(double));
    }

    if (debug->T)  cudaMemcpy2D(mixture_h.T,   sizeof(double)*i2m, plan.mixture_h.T,   sizeof(double)*i2m, sizeof(double)*i2m, j2m, cudaMemcpyHostToHost);
    if (debug->p0) cudaMemcpy2D(mixture_h.p0,  sizeof(double)*i1m, plan.mixture_h.p0,  sizeof(double)*i1m, sizeof(double)*i1m, j1m, cudaMemcpyHostToHost);
    if (debug->fg) cudaMemcpy2D(mixture_h.f_g, sizeof(double)*i2m, plan.mixture_h.f_g, sizeof(double)*i2m, sizeof(double)*i2m, j2m, cudaMemcpyHostToHost);
    if (debug->vxy)cudaMemcpy2D(mixture_h.vx,  sizeof(double)*i2n, plan.mixture_h.vx,  sizeof(double)*i2n, sizeof(double)*i2n, j2m, cudaMemcpyHostToHost);
    if (debug->vxy)cudaMemcpy2D(mixture_h.vy,  sizeof(double)*i2m, plan.mixture_h.vy,  sizeof(double)*i2m, sizeof(double)*i2m, j2n, cudaMemcpyHostToHost);

    if (debug->bubbles)
    {
        cudaMemcpy(bubbles_h.pos,  plan.bubbles_h.pos,  sizeof(double2)*numBubbles, cudaMemcpyHostToHost);
        cudaMemcpy(bubbles_h.R_t,  plan.bubbles_h.R_t,  sizeof(double)*numBubbles,  cudaMemcpyHostToHost);
        cudaMemcpy(bubbles_h.PG_p, plan.bubbles_h.PG_p, sizeof(double)*numBubbles,  cudaMemcpyHostToHost);
    }


    ofstream out_file;
    stringstream out;
    string out_dir = target_dir;
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

    if (debug->T)  free(mixture_h.T);
    if (debug->p0) free(mixture_h.p0);
    if (debug->fg) free(mixture_h.f_g);
    if (debug->vxy)free(mixture_h.vx);
    if (debug->vxy)free(mixture_h.vy);

    if (debug->bubbles)
    {
        free(bubbles_h.pos);
        free(bubbles_h.R_t);
        free(bubbles_h.PG_p);
    }
}
