export WANDB_ENTITY=mnoukhov
export WANDB_PROJECT=bpo
module load python/3.10
module load cuda/12.4.1/cudnn/8.9
source env/bin/activate
# mkdir $SLURM_TMPDIR/$SLURM_JOB_ID
# git clone --filter=blob:none --no-checkout $(GITHUB_REPO) $(_WORKDIR)
# git clone
# mkdir -p results/
