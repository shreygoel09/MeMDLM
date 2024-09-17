# to run:
# nohup bash run_memdlm.sh > memdlm_outs.out 2> memdlm_errors.err &

# run fine-tuning script
#export NCCL_SHM_DISABLE=1
CUDA_VISIBLE_DEVICES=6,7 /usr/bin/python3 main.py