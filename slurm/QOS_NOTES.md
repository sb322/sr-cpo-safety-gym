# Wulver QoS Notes

For the `ag2682` account on Wulver, SLURM launchers in this repository should use:

```bash
#SBATCH --partition=gpu
#SBATCH --account=ag2682
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
```

`qos=standard` is the normal project QoS for short smoke jobs, probe runs, and the current 50-epoch GPU sweeps. Avoid `priority`, `low`, `gpu_long`, `debug`, or an omitted QoS unless the cluster administrator explicitly asks for it; those settings can leave jobs pending for hours under reasons such as `Priority`, `Resources`, or `QOSGrpRunMinutesLimit`.
