# Simple benchmarking with NSight
nsys profile -o nsys_results/small python benchmark.py --configs small --num-warmups 5 --end-to-end --log-level INFO

python benchmarkv2.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
	--mode grad \
    --dtype fp16 / bf16 / fp32


# Annotated profiling with NSight. Following scripts are quite comprehensive,
# as they annotate forward, backward, optimizer step and also the attention operation.
nsys profile -o nsys_results/gpt_large_train --force-overwrite true \
python nsys_profile.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
    --annotate_attention \
	--mode train

nsys profile -o nsys_results/gpt_large_grad --force-overwrite true \
python nsys_profile.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
    --annotate_attention \
	--mode grad

nsys profile -o nsys_results/gpt_large_forward --force-overwrite true \
python nsys_profile.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
    --annotate_attention \
	--mode forward