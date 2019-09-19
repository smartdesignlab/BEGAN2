curr_date=`date +%y-%m-%d-%H-%M`
log_dir=./logs
mkdir -p ${log_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 main.py --n_z=128 --n_batch=16 --data_dir=/home/ubuntu/app1/input --dataset=wheeldesign --n_epoch=40000 --n_save_img_step=100 --n_save_log_step=30 --log_dir=${log_dir} > ${log_dir}/${curr_date}_log.txt &

sleep 1
tail -100f ${log_dir}/${curr_date}_log.txt
