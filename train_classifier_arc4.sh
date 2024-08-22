#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=12:00:00

#Request GPU
#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
module load cuda

python ./train_classifier.py
