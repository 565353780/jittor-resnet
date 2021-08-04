import jittor as jt
from PIL import ImageDraw, ImageFont
import numpy as np
import pickle
from PIL import Image
import cv2
import time
import os
from jittor.models import Resnet50, Resnet18
jt.flags.use_cuda = 1

# Load model checkpoint
experiment_id = "pretrain_model" # set your experiment id
model_path = os.path.join('tensorboard', experiment_id, 'model_best.pkl')
model_path = ('resnet50.pth')
#  params = pickle.load(open(model_path, "rb"))
model = Resnet50(pretrained=True)
#  model.load_parameters(params)
model.eval()
print(f'[*] Load model {model_path} success')

if __name__ == '__main__':
    img_dir = 'sample_images'
    try:
        os.makedirs('result/')
    except:
        print('Destination dir exists')
        pass
    time_list = []
    for i in range(10):
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            x = jt.random([10, 3, 224, 224])
            sta = time.time()
            ret = model(x)
            time_spend = time.time() - sta
            print("Once detect cost time:", time_spend)
            time_list.append(time_spend)
            #ret.save(os.path.join('result', img_name))

    avg_time = 0
    skip_num = 5
    skip_index = -1
    for time in time_list:
        skip_index += 1
        if skip_index < skip_num:
            continue
        avg_time += time
    avg_time /= (len(time_list) - skip_num)
    print("average time spend = ", avg_time)

