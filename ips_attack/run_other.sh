#!/bin/bash 
dataset=IPS #IPS, Splice
num=1
export CUDA_VISIBLE_DEVICES=$num
num_seqs=500
max_iter=7
method=bayesian  #greedy, bayesian
# kernel=hamming-cos-linear #hamming-cos-linear-rbf-rqk-matern-periodic-hik
for kernel in hamming-hik;do
    for threshold in 1;do
        for seed in 1;do
            if [ "$method" == "bayesian" ]; then
                dir="${kernel}-BO/${max_iter}_${threshold}/${num_seqs}/seed$seed"
            else
                kernel=none
                dir="${method}/$num_seqs/seed$seed"
            fi
            case $dataset in
                "IPS")
                    command="python attack_codes/attack.py classification --kernel $kernel --threshold $threshold --max_iter $max_iter --method $method --seed $seed --sidx 0 --num_seqs $num_seqs --working_folder datasets/clas_ec/clas_ec_ec50_level0 --max_patience 50 --Dataset IPS --pkl_dir $dir"
                    ;;
                "Splice")
                    command="python attack_codes/attack.py classification --kernel $kernel --threshold $threshold --max_iter $max_iter --method $method --seed $seed --sidx 0 --num_seqs $num_seqs --working_folder datasets/clas_ec/clas_ec_ec50_level0 --max_patience 50 --Dataset Splice --pkl_dir $dir"
                    ;;
            esac

            $command
            echo "$command" >> "$dir.txt"
            echo "$command"
        done
    done
done