# to run:
# nohup bash run_memdlm.sh > memdlm_outs.out 2> memdlm_errors.err &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 main.py