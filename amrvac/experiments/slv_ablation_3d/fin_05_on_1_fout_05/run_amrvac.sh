#!/bin/bash

#SBATCH -J run_amrvac             # Job name
#SBATCH -o run_amrvac.o%j         # Name of stdout output file
#SBATCH -e run_amrvac.e%j         # Name of stderr error file
#SBATCH -p cpu                # Queue (partition) name: normal or development
#SBATCH -N 1                  # Total # of nodes
#SBATCH --tasks-per-node 224   # Number of MPI tasks per node.
#SBATCH -t 3-23:59:59            # Run time (day-hh:mm:ss)

source /etc/profile.d/modules.sh

module load gcc/10.5.0-gcc-11.5.0-zanhivz
module load openmpi/4.1.6-gcc-10.5.0-o4iq6ut
module load perl/5.38.0-gcc-10.5.0-5sgdej3
module load python/3.11.6-gcc-10.5.0-dqhagsy

module list

echo "Using GCC: $(gcc --version | head -n1)"
echo "Using MPI: $(mpirun --version | head -n1)"

export AMRVAC_DIR=$HOME/repos/amrvac
export PATH=$PATH:.:$AMRVAC_DIR:$AMRVAC_DIR/tools

mpirun -np 224 ./amrvac -i amrvac.par
