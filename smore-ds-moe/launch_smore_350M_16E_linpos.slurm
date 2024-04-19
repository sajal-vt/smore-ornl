#!/bin/bash

#SBATCH -A STF218
#SBATCH -J SMORE-350M-E16-16N-linpos 
#SBATCH -o logs/smore-350M-E16-16n-linpos-%j.o
#SBATCH -e logs/smore-350M-E16-16n-linpos-%j.e
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 16
#SBATCH -d singleton

set +x
source /lustre/orion/world-shared/stf218/sajal/miniconda3-frontier/bin/activate
conda activate /lustre/orion/stf006/world-shared/irl1/flash2
#conda activate /lustre/orion/stf218/world-shared/sajal/flash2-moe

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu
module load rocm/5.7.0

module load PrgEnv-gnu

export HSA_DISABLE_CACHE=1

export ROCM_HOME=/opt/rocm-5.7.0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=INFO
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

export CHECKPOINT_PATH=checkpoints/tuning/smore-350M-E16-linpos
export VOCAB_FILE=science-vocab.json
export MERGE_FILE=science-merges.txt
export DATA_PATH=/lustre/orion/world-shared/stf218/sajal/science-data/all_text_document


DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs


## GPT-3 350M
MODEL_SIZE=0.35
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
LR=2.0e-4
MIN_LR=2e-06
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
TRAIN_TOKENS=300000000000
# TRAIN_TOKENS=330000000000

## TRAIN_ITERS is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=375000000
# LR_DECAY_TOKENS=260000000000
LR_DECAY_TOKENS=300000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
#BATCH_SIZE=8

## Model parallelism, 1 is no MP
## Currently MoE models have divergence issue when MP > 1.
MP_SIZE=1

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
NUM_GPUS=128
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
# EP_SIZE=1
EP_SIZE=16

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
LR=1.2e-4
MIN_LR=1.0e-6

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=5
EVAL_ITERS=5
EVAL_INTERVAL=100
SAVE_INTERVAL=20

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="true"
# ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}-test"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR} 
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
#CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# USE_INTERNAL_DATA="true"
USE_INTERNAL_DATA="false"

if [ "${USE_INTERNAL_DATA}" = "true" ]; then
    ## The internal data is only accessible within Microsoft
    ## For cluster Azure-EastUS-V100-32GB-4, Azure-WestUS3-A100
    # BASE_DATA_PATH=/vc_data/Megatron-LM/data
    # DATA_HOME="/vc_data/pile-cc1-cc2-shuf"
    ## For cluster Lab-RR1-V100
    BASE_DATA_PATH=/data/Megatron-LM/data
    DATA_HOME="/turing-ssd/users/conglli/data/pile-cc1-cc2-shuf"
    ## For cluster Azure-CentralUS-A100
    # BASE_DATA_PATH=/data/Megatron-LM/data
    # DATA_HOME=/vc_data_1/users/amawa/blended

    VOCAB_PATH=${BASE_DATA_PATH}/science-vocab.json
    MERGE_PATH=${BASE_DATA_PATH}/science-merges.txt
    ARX="${DATA_HOME}/ArXiv_ftfy_cleaned_id_shuf_text_document"
    BC2="${DATA_HOME}/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
    B3="${DATA_HOME}/Books3_ftfy_cleaned_id_shuf_text_document"
    CC2020="${DATA_HOME}/CC-2020-50_id_cleaned_shuf_text_document"
    CC2021="${DATA_HOME}/CC-2021-04_id_cleaned_shuf_text_document"
    GIT="${DATA_HOME}/Github_ftfy_id_shuf_text_document"
    GUT="${DATA_HOME}/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document"
    NIH="${DATA_HOME}/NIH_ExPorter_ftfy_id_shuf_text_document"
    OWT2="${DATA_HOME}/OpenWebText2_ftfy_cleaned_id_shuf_text_document"
    PCC="${DATA_HOME}/Pile-CC_id_cleaned_shuf_text_document"
    PM="${DATA_HOME}/PubMed_Abstracts_ftfy_id_shuf_text_document"
    RN="${DATA_HOME}/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document"
    SE="${DATA_HOME}/StackExchange_ftfy_id_shuf_text_document"
    ST="${DATA_HOME}/stories_dedup0.7_shuf_cleaned_shuf_text_document"
    WIK="${DATA_HOME}/Wikipedia_en_ftfy_id_shuf_text_document"
    DATA_BLEND="0.14336 ${B3} 0.08962 ${RN} 0.19336 ${OWT2} 0.05689 ${SE} \
    0.00859 ${ST} 0.02897 ${PM} 0.04771 ${WIK} 0.00873 ${GUT} 0.01007 ${BC2} \
    0.00208 ${NIH} 0.13017 ${CC2020} 0.09446 ${PCC} 0.15652 ${CC2021} \
    0.01359 ${ARX} 0.01588 ${GIT}"
else
    VOCAB_PATH=science-vocab.json
    MERGE_PATH=science-merges.txt
    # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    DATA_BLEND=/data/the_pile_public_merged_nopreprocessing/pile_text_document
fi
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH}"
        # --data-impl mmap"
        
megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR} \
	--moe-train-capacity-factor 1.25 \
	--expert-interval 2 \
	--topk 2 \
	--load ${CHECKPOINT_PATH}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

template_json="ds_config_gpt_TEMPLATE.json"
#template_json="ds_config_gpt_Zero2_TEMPLATE.json"
config_json="ds_config_gpt_Zero0_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
	  > ${config_json}

		    #--deepspeed_config ${config_json} \
deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --no-pipeline-parallel"
		    #--pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

#source export_DDP_vars.sh
srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest python pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} --master-addr=$MASTER_ADDR --use-flash-attn
#echo ${run_cmd}
#eval ${run_cmd}
set +x
