output_dir="./nohup/bayes/splice"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

CUDA_VISIBLE_DEVICES=1 python attack_codes/attack.py classification --method bayesian --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level0 --max_patience 50 --Dataset Splice > ./nohup/bayes/splice/seed_0.out

CUDA_VISIBLE_DEVICES=1 python attack_codes/attack.py classification --method bayesian --seed 1 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level0 --max_patience 50 --Dataset Splice > ./nohup/bayes/splice/seed_1.out

CUDA_VISIBLE_DEVICES=1 python attack_codes/attack.py classification --method bayesian --seed 2 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level0 --max_patience 50 --Dataset Splice > ./nohup/bayes/splice/seed_2.out