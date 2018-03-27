#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <math.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

#include "tools.h"  // Numerical tools


/* ---------------------------- COMMAND LINE INPUT -- -/

 -o: string
    Default: "out"
    Output file (without extension)
 
 -n: integer
    Default: 100
    Number of independent trials to simulate.
 
 -p: double
    Default: 1000.
    Population size cutoff (stop simulation when
    number of actively infected >= this number).
    
/* -------------------------------------------------- */


/* ------------------------------ GLOBAL VARIABLES -- */

double reactivation_rate  = 5.7e-5;  // rate of reactivation (per latent cell per day)
                                     // NOTE: death rate of latent cells is ignored because we consider large reservoirs
//double burst_rate         = 0.137;   // rate of actively-infected cell burst (per cell per day)
//double poisson_burst_size = 10.22;   // parameter for Poisson-distributed burst size (controls number of new infected)
//double death_rate         = 0.863;   // death rate of actively infected cells (per cell per day)
double mutation_rate      = 3e-5;    // mutation rate (per base per new infection event)
double recombination_rate = 1.4e-5;  // recombination rate (per base per new infection event) --> source: Neher and Leitner
double sequence_size      = 2600;    // approximate number of bases sequenced

double log10_r_mean   = -0.398;   // mean of log10 net growth rate (r) of rebound virus --> source: Hill
double log10_r_std    =  0.194;   // standard deviation of "" ""
double log10_rho_mean =  1.0;     // mean of log10 (rho) --> source: Hill (numbers chosen based on confidence interval)
double log10_rho_std  =  0.5;     // standard deviation of "" ""
double delta_mean     =  1.0;     // mean of net infected cell death rate --> source: Markowitz; in Hill this is fixed=1
double delta_std      =  0.3;     // standard deviation of "" ""
double log10_A_mean   =  1.755;   // mean of log10 number of reactivations per million latent cells (A) --> source: Hill
double log10_A_std    =  1.007;   // standard deviation of "" ""

double b_min_cutoff = 0.005; // minimum allowed value of random burst_rate
double d_min_cutoff = 0;     // minimum allowed value of random death_rate
double l_min_cutoff = 2.;    // minimum allowed value of random poisson_burst_size
double A_min_cutoff = 5.;    // minimum allowed value of random (reactivation_rate * 10^6)
double A_max_cutoff = 1e3;   // maximum allowed value of random (reactivation_rate * 10^6)

double latent_reservoir_size = 1e6;  // starting number of cells in the latent reservoir

/* -------------------------------------------------- */


/* -- generate parameters --------------------------- */

void generate_parameters(double r, double delta, double rho, double &b, double &d, double &l) {
    
    // Return underlying parameters b = burst_rate, d = death_rate, l = poisson_burst_size
    
    double dd = sqrt( (r * r) + ((2 * r * delta) * (rho - 1)) + ((delta * delta) * (rho - 1) * (rho + 3)));
    
    b  = ( r + delta + (rho * delta) - dd) / 2.;
    d  = (-r + delta - (rho * delta) + dd) / 2.;
    l  = ( r - delta + (rho * delta) + dd) / (2. * delta);

}

/* ---------------------------------- MAIN PROGRAM -- */

int main(int argc, char *argv[]) {
    
    // Process command line input
    
    std::string output_file       = "out";
    int         n_trials          = 100;
    double      population_cutoff = 1e3;
    
    for (int i=1;i<argc;i++) {
        
        if      (strcmp(argv[i],"-o")==0) { if (++i==argc) break; else output_file       = argv[i];              }
        else if (strcmp(argv[i],"-n")==0) { if (++i==argc) break; else n_trials          = strtoint(argv[i]);    }
        else if (strcmp(argv[i],"-p")==0) { if (++i==argc) break; else population_cutoff = strtodouble(argv[i]); }
        
        else printf("Unrecognized command! '%s'\n", argv[i]);
                
    }
    
    // Open output file
    
    FILE *out = fopen((output_file+".csv").c_str(),"w");
    fprintf(out,"trial,time,reactivation_rate,burst_rate,poisson_burst_size,death_rate,n,mutations\n");
    
    // Initialize RNG
    
    std::random_device rd;        // RNG for SGD
    std::mt19937       rng(rd()); // random seed
    
    //std::uniform_int_distribution<> uniform(0, cv.size()-1); // pick an observation uniformly at random
    //std::discrete_distribution<int> uniform(weights.begin(),weights.end());
    //std::bernoulli_distribution     binary(p);               // random 0/1 for mask, according to Bernoulli --> typecast to int
    
    std::normal_distribution<double> log10_A_distribution(log10_A_mean, log10_A_std);
    std::normal_distribution<double> log10_r_distribution(log10_r_mean, log10_r_std);
    std::normal_distribution<double> log10_rho_distribution(log10_rho_mean, log10_rho_std);
    std::normal_distribution<double> delta_distribution(delta_mean, delta_std);
    
    std::uniform_int_distribution<int> site_distribution(0, sequence_size);
    std::binomial_distribution<int> n_mutations_distribution(sequence_size, mutation_rate);
    std::binomial_distribution<int> n_breakpoints_distribution(sequence_size, recombination_rate);
    
    /* ------------------------------ loop trials -- */
    
    for (int t=0;t<n_trials;t++) {

        /* -- step 0 ------------ data structures -- */

        double latent_cells = latent_reservoir_size;
        double time         = 0;
        std::vector<double> active_cells;
        std::vector<std::vector<int> > active_cell_sequences;
        
        /* -- step 1 ----------- parameter choice -- */
        
        double A = pow(10., log10_A_distribution(rng));
        while (A<A_min_cutoff || A>A_max_cutoff) {
            A = pow(10., log10_A_distribution(rng));
        }
        double reactivation_rate = A / latent_reservoir_size;
        
        double b, d, l;
        double r     = pow(10., log10_r_distribution(rng));
        double rho   = pow(10., log10_rho_distribution(rng));
        double delta = delta_distribution(rng);
        generate_parameters(r, delta, rho, b, d, l);
        
        while (isnan(b) || b<b_min_cutoff || isnan(d) || d<d_min_cutoff || isnan(l) || l<l_min_cutoff) {
            r     = pow(10., log10_r_distribution(rng));
            rho   = pow(10., log10_rho_distribution(rng));
            delta = delta_distribution(rng);
            generate_parameters(r, delta, rho, b, d, l);
        }
        double burst_rate         = b;
        double death_rate         = d;
        double poisson_burst_size = l;
        
        std::poisson_distribution<int> burst_size_distribution(poisson_burst_size);

        /* -- step 2 ------------- run simulation -- */

        double n_active   = 0;
        double total_rate = 0;
        std::vector<double> action_table(3, 0);

        // DEBUG
        //int counter = 0;

        while (n_active<population_cutoff && time<100) {
        
            n_active = 0;
            for (int i=0;i<active_cells.size();i++) n_active += active_cells[i];
        
            //DEBUG
            //if (counter%1000==0) printf("\n%d\t%lf",counter,n_active);
            //counter++;
        
            action_table[0] = reactivation_rate * latent_cells;
            action_table[1] = death_rate * n_active;
            action_table[2] = burst_rate * n_active;
            
            total_rate = 0;
            for (int i=0;i<action_table.size();i++) total_rate += action_table[i];
            
            std::discrete_distribution<int> action_distribution(action_table.begin(), action_table.end());
            std::exponential_distribution<double> delta_t_distribution(total_rate);
            
            int action  = action_distribution(rng);
            time       += delta_t_distribution(rng);
        
            /* --------------------- reactivation -- */
            if (action==0) {
            
                bool reactivated = false;
                for (int i=0;i<active_cells.size() && !reactivated;i++) {
                
                    int n_mutations = 0;
                    for (int j=0;j<active_cell_sequences[i].size();j++) {
                        n_mutations += active_cell_sequences[i][j];
                    }
                    
                    if (n_mutations==0) {
                        active_cells[i] += 1.;
                        reactivated      = true;
                    }
                
                }
                
                if (!reactivated) {
                    active_cells.insert(active_cells.begin(), 1.);
                    active_cell_sequences.insert(active_cell_sequences.begin(), std::vector<int>(sequence_size,0));
                }
            
            }
    
            /* ----------------------- cell death -- */
            // Choose a random actively-infected cell to die
            
            else if (action==1) {
            
                std::discrete_distribution<int> death_distribution(active_cells.begin(), active_cells.end());
                int death            = death_distribution(rng);
                active_cells[death] -= 1;
                
                if (active_cells[death]==0) {
                    active_cells.erase(active_cells.begin() + death);
                    active_cell_sequences.erase(active_cell_sequences.begin() + death);
                }
                
            }
    
            /* ----------------------- cell burst -- */
            // Choose a random actively-infected cell to burst with Poisson burst size
            // Each infection can generate new mutations
            
            else if (action==2) {
            
                std::discrete_distribution<int> burst_distribution(active_cells.begin(), active_cells.end());
                
                int burst      = burst_distribution(rng);
                int burst_size = burst_size_distribution(rng);
                
                if (burst_size>0) {
                
                    // Add new actively-infected cells to the pool
                    
                    for (int i=0;i<burst_size;i++) {
                    
                        // The new sequence starts as a copy of the old
                        std::vector<int> new_sequence(active_cell_sequences[burst]);
                        
                        // Recombination: recombine with a random sequence from the set of active cells
                        // Choose breakpoints uniformly at random; by convention the "parent" sequence starts at 0
                        int n_breakpoints = n_breakpoints_distribution(rng);
                        if (n_breakpoints>0) {
                        
                            std::discrete_distribution<int> partner_idx_distribution(active_cells.begin(), active_cells.end());
                            int partner_idx = partner_idx_distribution(rng);
                            
                            std::vector<int> breakpoints;
                            if (n_breakpoints%2==1) breakpoints.push_back(sequence_size);
                            
                            for (int j=0;j<n_breakpoints;j++) {
                            
                                int  breakpoint = site_distribution(rng);
                                bool inserted   = false;
                                for (int k=0;k<breakpoints.size() && !inserted;k++) {
                                    if (breakpoint<breakpoints[k]) {
                                        breakpoints.insert(breakpoints.begin() + k, breakpoint);
                                        inserted = true;
                                    }
                                }
                                if (!inserted) breakpoints.push_back(breakpoint);
                                
                            }
                            
                            int index = 0;
                            for (int j=0;j<breakpoints.size();j=j+2) {
                                for (int k=breakpoints[j];k<breakpoints[j+1];k++) {
                                    new_sequence[k] = active_cell_sequences[partner_idx][k];
                                }
                            }
                            
                        }
                        
                        // Mutation: add de novo mutations
                        int n_mutations = n_mutations_distribution(rng);
                        for (int j=0;j<n_mutations;j++) {
                            int site = site_distribution(rng);
                            new_sequence[site] = 1 - new_sequence[site]; // binary mutation approximation
                        }
                        
                        // Check for existence of sequence in the list
                        std::vector<std::vector<int> >::iterator it;
                        it = std::find(active_cell_sequences.begin(), active_cell_sequences.end(), new_sequence);
                        
                        if (it==active_cell_sequences.end()) {
                            active_cells.push_back(1.);
                            active_cell_sequences.push_back(new_sequence);
                        }
                        else {
                            int index = std::distance(active_cell_sequences.begin(), it);
                            active_cells[index] += 1.;
                        }
                    
                    }
                    
                    // Remove the burst cell
                    active_cells[burst] -= 1.;
                    if (active_cells[burst]==0) {
                        active_cells.erase(active_cells.begin() + burst);
                        active_cell_sequences.erase(active_cell_sequences.begin() + burst);
                    }
                    
                }
            
            }

        }

        /* -- step 3 ---------------- save output -- */

        // Reorganize according to number of mutations
        std::vector<int> cell_counts;
        std::vector<int> cell_mutations;
        for (int i=0;i<active_cells.size();i++) {
            int n_mut = 0;
            for (int j=0;j<active_cell_sequences[i].size();j++) n_mut += active_cell_sequences[i][j];
            bool counted = false;
            for (int j=0;j<cell_mutations.size() && !counted;j++) {
                if (cell_mutations[j]==n_mut) {
                    cell_counts[j] += active_cells[i];
                    counted = true;
                }
            }
            if (!counted) {
                bool inserted = false;
                for (int j=0;j<cell_mutations.size() && !inserted;j++) {
                    if (n_mut<cell_mutations[j]) {
                        cell_mutations.insert(cell_mutations.begin() + j, n_mut);
                        cell_counts.insert(cell_counts.begin() + j, active_cells[i]);
                        inserted = true;
                    }
                }
                if (!inserted) {
                    cell_mutations.push_back(n_mut);
                    cell_counts.push_back(active_cells[i]);
                }
            }
        }
        
        // Order entries by number of mutations

        for (int i=0;i<cell_counts.size();i++) {
            fprintf(out,"%d,%lf,%lf,%lf,%lf,%lf", t, time, reactivation_rate, burst_rate, poisson_burst_size, death_rate);
            fprintf(out,",%d,%d\n", cell_counts[i], cell_mutations[i]);
        }
        fflush(out);
        
    }
        
    return EXIT_SUCCESS;
    
}
