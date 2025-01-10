#!/bin/bash 
model=mr/cnn # mr/xlnet, ag/lstm, ag/bert, mr/bert, mr/lstm, ag/cnn, mr/cnn
num=2
export CUDA_VISIBLE_DEVICES=$num
examples=500
max_iter=7
threshold=0.7
recipe=pso-wordnet #textfooler,pso,pwws,bayesattack-wordnet
# hamming-cos-linear-rbf-rqk-matern-periodic-hik
for kernel in hamming-cos-linear;do
    for seed in 0 1 2;do
        if [ "$recipe" == "bayesattack-wordnet" ]; then
            dir="NEW_RESULTS/$model/${kernel}-BO/${max_iter}_${threshold}/${examples}/seed$seed"
        else
            kernel=none
            dir="NEW_RESULTS/$model/${recipe}/$examples/seed$seed"
        fi
        mkdir -p "$dir"
        if [ ! -e "$dir.txt" ]; then
            touch "$dir.txt"
        fi

        case $model in
            "mr/xlnet")
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model xlnet-base-cased-mr --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 100 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "ag/lstm") 
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model lstm-ag-news --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "ag/bert")
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model bert-base-uncased-ag-news --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "mr/bert")
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model bert-base-uncased-mr --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "mr/lstm")
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model lstm-mr --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "ag/cnn") 
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model cnn-ag-news --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
            "mr/cnn")
                command="python -m textattack attack --silent --shuffle --shuffle-seed 0 --random-seed $seed --recipe $recipe --transformation word-swap-wordnet --model cnn-mr --sidx 0 --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50 --num-examples $examples --kernel $kernel --threshold $threshold --max-iter $max_iter --pkl-dir $dir"
                ;;
        esac

        $command
        echo "$command" >> "$dir.txt"
        echo "$command"
    done
done