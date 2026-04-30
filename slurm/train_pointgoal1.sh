#!/usr/bin/env bash
#SBATCH --job-name=sr_cpo_pointgoal
#SBATCH --output=pointgoal1.%j.out
#SBATCH --error=pointgoal1.%j.err
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

echo "SR-CPO MJX PointGoal-style launcher is not implemented yet."
exit 1
