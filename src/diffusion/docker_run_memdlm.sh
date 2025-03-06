# to run:
# nohup bash run_memdlm.sh > memdlm_outs.out 2> memdlm_errors.err &

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 /usr/bin/python3 main.py