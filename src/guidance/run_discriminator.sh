# to run:
# nohup bash run_discriminator.sh > discriminator_outs.out 2> discriminator_errors.err &

CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python3 main.py