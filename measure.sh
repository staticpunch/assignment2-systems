for warmups in 0 5; do
    for end_to_end in "" "--end-to-end"; do
        echo "Running: warmups=$warmups, end_to_end=${end_to_end:-False}"
        
        python cs336_systems/measure.py \
            --num-warmups $warmups \
            $end_to_end \
            --quiet \
            --log-level INFO
    done
done
