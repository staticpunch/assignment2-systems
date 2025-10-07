HEAD_DIMS=(16 32 64 128)
SEQ_LENGTHS=(256 1024 4096 8192 16384)
CSV_FILE="results/compiled_attention.csv"

rm  "$CSV_FILE"
echo "head_dim,seq_length,mean_ms,std_ms,min_ms,max_ms" > "$CSV_FILE"

for head_dim in "${HEAD_DIMS[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "Running: head_dim=$head_dim, seq_len=$seq_len"
        python benchmark_attn.py \
            --n-queries "$seq_len" \
            --n-keys "$seq_len" \
            --head-dim "$head_dim" \
            --csv "$CSV_FILE" \
	    --compile
    done
done
