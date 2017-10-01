# Simple viral rebound model based on Hill et al.
# We model the rebound of HIV from the latent reservoir
# and track the number of mutations accumulated over time

import sys
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

latent_reservoir_size = 1e6  # starting number of cells in the latent reservoir


###### Main functions ######


def usage():
    print("")


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
    f.write('trial,time,n,mutations\n')
    
    for t in range(n_trials):
    
        print_update(t, n_trials)   # status check

        # INITIALIZATION - DEFINE DATA STRUCTURES

        latent_cells          = latent_reservoir_size
        active_cells          = np.array([])
        active_cell_mutations = np.array([])
        time                  = 0

        # STOCHASTIC SIMULATION

        while np.sum(active_cells)<population_cutoff:
        
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
                    
                    # GROUP NEW ACTIVE BY NUMBER OF MUTATIONS FIRST?
                    #new_active    = [1.]
                    #new_mutations = [n_mutations[0]]
                    #for i in range(1, burst_size):
                    #    if n_mutations[i] in new_mutations:
                    #        new_active[new_mutations.index(n_mutations[i])] += 1.
                    #    else:
                    #        new_active.append(1.)
                    #        new_mutations.append(n_mutations[i])
                    #for i in range(new_active):
                    #    if new_mutations[i] in active_cell_mutations:
                    #        idx                = np.where(active_cell_mutations==new_mutations[i])[0][0]
                    #        active_cells[idx] += new_active[i]
                    #    else:
                    #        active_cells          = np.append(active_cells, new_active[i])
                    #        active_cell_mutations = np.append(active_cell_mutations, new_mutations[i])
                    
        # SAVE OUTPUT

        for i in range(len(active_cells)):
            f.write('%d,%lf,%d,%d\n' % (t, time, active_cells[i], active_cell_mutations[i]))
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

