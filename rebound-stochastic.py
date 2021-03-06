# Simple viral rebound model based on Hill et al.,
# PNAS 2014 (https://doi.org/10.1073/pnas.1406663111)
# We model the rebound of HIV from the latent reservoir
# and track the number of mutations accumulated over time

# This version generates stochastic parameters
# (reactivation, death, and burst rates) using parameter
# ranges from Hill et al. and Markowitz et al., J Virol
# 2003 (https://doi.org/10.1128/JVI.77.8.5037-5038.2003)

# Specifically, we choose the net growth rate (r), ratio
# of variance to mean number of offspring for an infected
# cell (rho), total infected cell death rate (delta), and
# number of latent reactivations per day per million (A)
# to be random from the prescribed distributions, then work
# back to find consistent model parameters

import sys
import argparse
import numpy as np                          # numerical tools
from copy import deepcopy                   # deepcopy copies a data structure without any implicit references
from timeit import default_timer as timer   # timer for performance


###### Global parameters ######

reactivation_rate  = 5.7e-5  # rate of reactivation (per latent cell per day)
# NOTE: death rate of latent cells is ignored because we consider large reservoirs
burst_rate         = 0.137   # rate of actively-infected cell burst (per cell per day)
poisson_burst_size = 10.22   # parameter for Poisson-distributed burst size (controls number of new infected)
death_rate         = 0.863   # death rate of actively infected cells (per cell per day)
mutation_rate      = 3e-5    # mutation rate (per base per new infection event)
sequence_size      = 2600    # approximate number of bases sequenced

log10_r_mean   = -0.398   # mean of log10 net growth rate (r) of rebound virus --> source: Hill
log10_r_std    =  0.194   # standard deviation of "" ""
log10_rho_mean =  1.0     # mean of log10 (rho) --> source: Hill (numbers chosen based on confidence interval)
log10_rho_std  =  0.5     # standard deviation of "" ""
delta_mean     =  1.0     # mean of net infected cell death rate --> source: Markowitz; in Hill this is fixed=1
delta_std      =  0.3     # standard deviation of "" ""
log10_A_mean   =  1.755   # mean of log10 number of reactivations per million latent cells (A) --> source: Hill
log10_A_std    =  1.007   # standard deviation of "" ""

b_min_cutoff = 0.005 # minimum allowed value of random burst_rate
d_min_cutoff = 0     # minimum allowed value of random death_rate
l_min_cutoff = 2.    # minimum allowed value of random poisson_burst_size
A_min_cutoff = 5.    # minimum allowed value of random (reactivation_rate * 10^6)
A_max_cutoff = 1e3   # maximum allowed value of random (reactivation_rate * 10^6)

latent_reservoir_size = 1e6  # starting number of cells in the latent reservoir


###### Main functions ######


def usage():
    print("")

def get_parameters(r, delta, rho):
    """ Return underlying parameters b = burst_rate, d = death_rate, l = poisson_burst_size """
    dd = np.sqrt(r**2 + (2 * r * delta) * (rho - 1) + delta**2 * (rho - 1) * (rho + 3))
    b  = ( r + delta + (rho * delta) - dd) / 2.
    d  = (-r + delta - (rho * delta) + dd) / 2.
    l  = ( r - delta + (rho * delta) + dd) / (2. * delta)
    return b, d, l

def main(verbose=False):
    """ Simulate the outgrowth and rebound of latent virus and save the results to a CSV file. """
    
    # Run multiple trials and save all data to file
    
    parser = argparse.ArgumentParser(description='Simulate viral rebound')
    parser.add_argument('-o',   type=str,   default='rebound', help='output file (without extension)')
    parser.add_argument('-n',   type=int,   default=100,       help='number of independent trials to simulate')
    parser.add_argument('-p',   type=float, default=1e3,       help='population cutoff (stop trial when actively infected >= this number)')
    
    arg_list = parser.parse_args(sys.argv[1:])
    
    output_file       = arg_list.o
    n_trials          = arg_list.n
    population_cutoff = arg_list.p
    
    start = timer()
    
    f = open(output_file+'.csv', 'w')
    f.write('trial,time,reactivation_rate,burst_rate,poisson_burst_size,death_rate,n,mutations\n')
    
    for t in range(n_trials):
        
        print_update(t, n_trials)   # status check
        
        # INITIALIZATION - DEFINE DATA STRUCTURES
        
        latent_cells          = latent_reservoir_size
        active_cells          = np.array([])
        active_cell_mutations = np.array([])
        time                  = 0
        
        # GENERATE STOCHASTIC PARAMETERS
        
        A = 10.**np.random.normal(log10_A_mean, log10_A_std)
        while A<A_min_cutoff or A>A_max_cutoff:
            A = 10.**np.random.normal(log10_A_mean, log10_A_std)
        reactivation_rate = A / latent_reservoir_size
        
        r     = 10.**np.random.normal(log10_r_mean, log10_r_std)
        rho   = 10.**np.random.normal(log10_rho_mean, log10_rho_std)
        delta = np.random.normal(delta_mean, delta_std)
        b, d, l = get_parameters(r, delta, rho)
        while np.isnan(b) or b<b_min_cutoff or np.isnan(d) or d<d_min_cutoff or np.isnan(l) or l<l_min_cutoff:
            r     = 10.**np.random.normal(log10_r_mean, log10_r_std)
            rho   = 10.**np.random.normal(log10_rho_mean, log10_rho_std)
            delta = np.random.normal(delta_mean, delta_std)
            b, d, l = get_parameters(r, delta, rho)
        burst_rate         = b
        death_rate         = d
        poisson_burst_size = l
        
        # STOCHASTIC SIMULATION
        
        while np.sum(active_cells)<population_cutoff and time<100:
            
            n_active     = np.sum(active_cells)
            action_table = np.array([reactivation_rate * latent_cells, death_rate * n_active, burst_rate * n_active])
            total_rate   = np.sum(action_table)
            
            action       = np.random.choice(range(3), p = action_table / total_rate)
            delta_t      = np.random.exponential(1. / total_rate)   # numpy uses INVERSE of rate
            time        += delta_t
            
            # Latent reactivation
            if action==0:
                reactivated = False
                for i in range(len(active_cells)):
                    if active_cell_mutations[i]==0:
                        active_cells[i] += 1.
                        reactivated      = True
                        break
                if not reactivated:
                    active_cells          = np.append(active_cells, 1.)
                    active_cell_mutations = np.append(active_cell_mutations, 0)
        
            # Active cell death
            # Choose a random actively-infected cell to die
            elif action==1:
                death                = np.random.choice(range(len(active_cells)), p = active_cells / n_active)
                active_cells[death] -= 1
                if active_cells[death]==0:
                    active_cells          = np.delete(active_cells, death)
                    active_cell_mutations = np.delete(active_cell_mutations, death)

            # Active cell burst
            # Choose a random actively-infected cell to burst with Poisson burst size
            # Each infection can generate new mutations
            elif action==2:
                burst      = np.random.choice(range(len(active_cells)), p = active_cells / n_active)
                burst_size = np.random.poisson(poisson_burst_size)
                if burst_size>0:
                    # Add new actively-infected cells to the pool
                    n_mutations = np.random.binomial(sequence_size, mutation_rate, size = burst_size) + active_cell_mutations[burst]
                    for i in range(len(n_mutations)):
                        if n_mutations[i] in active_cell_mutations:
                            idx                = np.where(active_cell_mutations==n_mutations[i])[0][0]
                            active_cells[idx] += 1.
                                else:
                                    active_cells          = np.append(active_cells, 1.)
                                    active_cell_mutations = np.append(active_cell_mutations, n_mutations[i])
                                # Remove the burst cell
                                active_cells[burst] -= 1.
                                if active_cells[burst]==0:
                                    active_cells          = np.delete(active_cells, burst)
                                    active_cell_mutations = np.delete(active_cell_mutations, burst)
        
        # SAVE OUTPUT
        
        for i in range(len(active_cells)):
            f.write('%d,%lf,%lf,%lf,%lf,%lf' % (t, time, reactivation_rate, burst_rate, poisson_burst_size, death_rate))
            f.write(',%d,%d\n' % (active_cells[i], active_cell_mutations[i]))
        f.flush()

# End and output total time

f.close()
    
    end = timer()
    print('\nTotal time: %lfs, average per cycle %lfs' % ((end - start),(end - start)/float(n_trials)))


def print_update(current, end, bar_length=20):
    """ Print an update of the simulation status. h/t Aravind Voggu on StackOverflow. """
    
    percent = float(current) / end
    dash    = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
    space   = ''.join([' ' for k in range(bar_length - len(dash))])
    
    sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
    sys.stdout.flush()


if __name__ == '__main__': main()
