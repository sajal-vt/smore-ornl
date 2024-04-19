#!/bin/bash

#SBATCH -A STF218
#SBATCH -J tutel-moe 
#SBATCH -o logs/tutel-smore-1.3b-%j.o
#SBATCH -e logs/tutel-smore-1.3b-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 4

set +x
source /lustre/orion/world-shared/stf218/sajal/miniconda3-frontier/bin/activate
conda activate /lustre/orion/stf218/world-shared/sajal/mtds/Megatron-DeepSpeed-ds/env-flash

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu
module load gcc/11.2.0
module load rocm/5.4.0

export ROCM_HOME=/opt/rocm-5.4.0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# setup env file
echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
echo "CPATH=$CPATH" >> .deepspeed_env
echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
echo "HF_HOME=$PWD/hfdata" >> .deepspeed_env
echo "ROCM_HOME=/opt/rocm-5.4.0" >> .deepspeed_env



scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

echo $ranks_per_node $gpus_per_rank $ranks_total
mkdir logs
mkdir logs/transformer

export CHECKPOINT_PATH=checkpoints/gpt2_345m
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt
export DATA_PATH=/lustre/orion/world-shared/stf218/sajal/mtds/gptdata/gpttext_article_document

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/sajal/software/aws-ofi-rccl/src/.libs/:${LD_LIBRARY_PATH}
export NCCL_NET_GDR_LEVEL=3
export FI_MR_CACHE_MONITOR=userfaultfd
#python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=<node-ip-0> -m tutel.examples.helloworld --batch_size=16
export PYTHON_EXE=/lustre/orion/stf218/world-shared/sajal/mtds/Megatron-DeepSpeed-ds/env-flash/bin/python
time srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest $PYTHON_EXE smore_tutel.py --batch_size=16 --dtype float16 --num_local_experts 0.49 --model_dim=2048 --hidden_size 8192 --num_layers 24 --num_steps 5 --num_tokens 2048

