#!/bin/bash
#
#SBATCH --job-name=tykfowe_zadanie
#SBATCH --output=tykfowe_zadanie.out
#SBATCH --error=tykfowe_zadanie.err
#SBATCH --partition=dziobas
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=20:00:00

python 11.1.1_moja_sieci_neuronowa.py
