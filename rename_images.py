import os
import datetime

path = './Extract/src/'
dt_now = datetime.datetime.now()
dt_now = dt_now.strftime("%Y%m%d%H%M%S")

f_list = os.listdir(path)
for idx, f in enumerate(f_list):
    src_name = path + f
    dist_name = path + '%s_%d.jpg'%(dt_now, idx)
    os.rename(src_name, dist_name) 
