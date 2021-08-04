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

    run_episode = 10
    time_list = []
    detected_num = 0
    total_num = run_episode * len(os.listdir(img_dir))
    for i in range(run_episode):
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            x = jt.random([10, 3, 224, 224])
            sta = time.time()
            ret = model(x)
            time_spend = time.time() - sta
            detected_num += 1
            print("\rDetect process:", detected_num, "/", total_num, "    ", end="")
            time_list.append(time_spend)
            #ret.save(os.path.join('result', img_name))
    print()

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

