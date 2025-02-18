# to run:
# nohup bash run_protgpt2.sh > protgpt2_outs.out 2> protgpt2_errors.err &

# run fine-tuning script
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 protgpt2_generate.py