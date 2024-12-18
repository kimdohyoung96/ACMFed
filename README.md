Preparation
1. Create conda environment
 conda create -n ACMFed python=3.8
 conda activate ACMFed

2. install depencies
 pip install -r requirements.txt

Run the code
3. Train model for each dataset. To produce the claimed results for SVHN dataset.

python train_main.py --dataset=SVHN \
	--model=simple-cnn \
	--unsup_num=9 \
	--batch_size=64 \
	--lambda_u=0.02 \
	--opt=sgd \
	--base_lr=0.03 \
	--unsup_lr=0.021 \
	--max_grad_norm=5 \
	--resume \
	--from_labeled \
	--rounds=1000 \
	--meta_round=3 \
	--meta_client_num=5 \
	--w_mul_times=6 \
	--sup_scale=100 \
	--dist_scale=1e4 \

 Evaluation
 Use the following command to generate the claimed results
 python test.py --dataset=SVHN \
	--batch_size=5 \
	--model=simple-cnn \

 For different datasets, please modify file path, arguments "dataset" and "model" correspondingly.
 
