# to run:
# nohup bash run_mdplm.sh > mdplm_outs.out 2> mdplm_errors.err &

# nohup bash run_mdplm.sh > mdplm_generate.out 2> mdplm_generate.err &

# run fine-tuning script
HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 mdlm_motif_benchmarking.py