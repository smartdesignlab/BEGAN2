import os, pprint, time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import scipy.misc
from glob import glob
from random import shuffle
from models import GeneratorCNN
from utils import save_image
from config import get_config
pp = pprint.PrettyPrinter()

def main():

    #load configuration
    conf, _ = get_config()
    
    def getRandomG(ckpt_prefix):

        anal_dir = os.path.join(conf.data_dir,"anal")
        if not os.path.exists(anal_dir):
            os.makedirs(anal_dir)
    
        n_step=0
        for i in range(3):
            z_test =np.random.normal(loc=0.0, scale=1.0, size=(conf.n_batch, conf.n_z)).astype(np.float32)
            g_im =sess.run(g_img,feed_dict={z:z_test})
            for j in range(conf.n_batch):
                save_path = os.path.join(anal_dir,str(conf.n_z)+'_'+ckpt_prefix+'_'+str(n_step)+'_anal_G.jpg')
                save_image(g_im[n_step%conf.n_batch].reshape(1, conf.n_img_out_pix, conf.n_img_out_pix, 1), [1,1],save_path)
                n_step+=1
        
    ckpt_dir=os.path.join(conf.load_path,"ckpt")
    ckpt_list=[]
    ckpt_files = os.listdir(ckpt_dir)
    for ckpt_file in ckpt_files:
        a,b,c,d=ckpt_file.split('_')
        d, _, _ = d.split('.')
        ckpt_prefix = a+"_"+b+"_"
        ckpt_name=c+"_"+d+".ckpt"
        full_ckpt_name = ckpt_prefix+ckpt_name

        if ckpt_list.__contains__(ckpt_prefix):
            continue

        ckpt_list.append(ckpt_prefix)

        if conf.is_gray :
            n_channel=1
        else:
            n_channel=3

        n_grid_row = int(np.sqrt(conf.n_batch))

        z = tf.random_uniform((conf.n_batch, conf.n_z), minval=-1.0, maxval=1.0)
        net_g, g_logits = GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse)
        g_img=tf.clip_by_value((net_g + 1)*127.5, 0, 255)
        # start session
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_dir,full_ckpt_name ))
        
        getRandomG(ckpt_prefix)
        sess.close()

if __name__ == '__main__':
    main()
