# Simple benchmarking with NSight
nsys profile -o nsys_results/small python benchmark.py --configs small --num-warmups 5 --end-to-end --log-level INFO

python benchmark.py \
	--config large \
	--num-steps 20 \
	--num-warmups 10 \
	--batch-size 1 \
	--sequence-length 128 \
    --mode grad \
    --memory \
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


# Benchmarking attention implementations
HEAD_DIMS=(16 32 64 128)
SEQ_LENGTHS=(256 1024 4096 8192 16384)
CSV_FILE="results/naive_attention.csv"

rm -f "$CSV_FILE"
echo "head_dim,seq_length,mean_ms,std_ms,min_ms,max_ms" > "$CSV_FILE"

for head_dim in "${HEAD_DIMS[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "Running: head_dim=$head_dim, seq_len=$seq_len"
        python benchmark_attn.py \
            --n-queries "$seq_len" \
            --n-keys "$seq_len" \
            --head-dim "$head_dim" \
            --csv "$CSV_FILE"
    done
done

echo "Results saved to $CSV_FILE"