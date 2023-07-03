#!/bin/sh

#SBATCH --job-name=train_vid_stream.sh
#SBATCH --output=/home/guests/nicolas_gossard/outputs/train_vid_stream-%A.out
#SBATCH --error=/home/guests/nicolas_gossard/IDP/errors/train_vid_stream-%A.err    #realtive or absolute path
#SBATCH --time=0-00:05:00   # maximum running time (format is day-hours:min:sec)
#SBATCH --gres=gpu:1   #number of GPUs if needed
#SBATCH --cpus-per-task=8  #number of CPUs (no more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (no more tha  126GB per GPU)

# load the python module
ml python/anaconda3
#activate the good environment 
source activate pyfl

python /home/guests/nicolas_gossard/IDP/train.py -c /home/guests/nicolas_gossard/IDP/modules/multiclass_classifier/config/X3D.yml

