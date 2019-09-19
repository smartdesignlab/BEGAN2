curr_date=`date +%y-%m-%d-%H-%M`
log_dir=./logs
mkdir -p ${log_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3
#/disk1/archive/began_2019/19-01-06-15-39
python3 anal_ckpt.py --load_dir=/home/ubuntu/BEGAN-tensorflow/logs/091440 --n_z=64 --n_batch=16  --n_epoch=4000 --n_save_img_step=600 --n_save_log_step=300 --log_dir=${log_dir} > ${log_dir}/${curr_date}_log.txt &

sleep 1
tail -100f ${log_dir}/${curr_date}_log.txt
