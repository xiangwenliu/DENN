#! /bin/bash
#data path
data_path='~/tensorflowproject/data/uci'
result_fold='result'


if [ ! -d ${data_path} ]; then
    echo "the path ${data_path} does not exist!!"
    exit
fi

cwd=$(cd `dirname $0`; pwd)
echo "${cwd}"
#result fold
if [ -d ${result_fold} ]; then
    echo "the result fold exist!!"
    exit
else
   echo "make a dir"
fi
resultpath="${cwd}/${result_fold}"
echo "result path is:${resultpath}"

for item in $(ls ${data_path})
do
    #echo "${item}"
    resultfilepath="${resultpath}/${item}"
    mkdir -p "${resultfilepath}"
    current_train_data_path="${data_path}/${item}"
    train_data="${current_train_data_path}/${item}"_py.dat""
    train_label="${current_train_data_path}/"labels_py.dat""
    test_folds="${current_train_data_path}/"folds_py.dat""
    validation_folds="${current_train_data_path}/"validation_folds_py.dat""
    if [ -f ${train_data} ] && [ -f ${train_label} ]; then
		for keep in 1.0 0.8
		do
			for branch in 2 4 8 16 32 64 128 256
			do
				python train_uci.py \
				  --data_set=${item} \
				  --model_name=dendritenet \
				  --data_dir=${train_data} \
				  --label_dir=${train_label} \
				  --test_folds=${test_folds} \
				  --validation_folds=${validation_folds} \
				  --result_dir=${resultfilepath} \
				  --branches=${branch} \
				  --layers=3 \
				  --learning_rate=0.01 \
				  --net_length=512 \
				  --keep_prob=${keep} \
				  --Normalization=No
			done
		done
    else
        echo "******The data file missing ******"
    fi
done

