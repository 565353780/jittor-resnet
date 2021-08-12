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

class ResNetDetector:
    def __init__(self):
        self.valid_net_depth = [18, 50]
        self.net_depth = None
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def reset(self):
        self.net_depth = None
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

    def startTimer(self):
        self.time_start = time.time()

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")

    def getAverageTime(self):
        if self.detected_num == 0:
            return -1

        return 1.0 * self.total_time_sum / self.detected_num

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def loadModel(self, net_depth, model_path=None):
        self.reset()
        self.net_depth = net_depth

        if self.net_depth not in self.valid_net_depth:
            print("Invalid net_depth!")
            return

        need_pretrained_param = model_path is None

        if self.net_depth == 18:
            self.model = Resnet18(pretrained=need_pretrained_param)
        elif self.net_depth == 50:
            self.model = Resnet50(pretrained=need_pretrained_param)

        if model_path is not None:
            if not os.path.exists(model_path):
                print("Model not exists!")
                return

            params = pickle.load(open(model_path, "rb"))
            self.model.load_parameters(params)

        self.model.eval()
        if model_path is None:
            print("Load model pretrained Resnet" + str(self.net_depth) + " success")
        else:
            print("Load model " + model_path + " success")
        self.model_ready = True
        return

    def detect(self, image):
        result = self.model(image)
        return result

    def test(self, image_folder_path, run_episode=10, timer_skip_num=5):
        if not self.model_ready:
            print("Model not ready yet, Please loadModel or check your model path first!")
            return

        if run_episode == 0:
            print("No detect run with run_episode=0!")
            return

        image_file_name_list = os.listdir(image_folder_path)

        if run_episode < 0:
            self.resetTimer()
            timer_skipped_num = 0

            while True:
                for image_file_name in image_file_name_list:
                    #  image_file_path = os.path.join(image_folder_path, image_file_name)
                    image = jt.random([10, 3, 224, 224])

                    self.startTimer()

                    result = self.detect(image)

                    if timer_skipped_num < timer_skip_num:
                        self.endTimer(False)
                        timer_skipped_num += 1
                    else:
                        self.endTimer()

                    print("\rNet: ResNet" + str(self.net_depth) +
                          "\tDetected: " + str(self.detected_num) +
                          "\tAvgTime: " + str(self.getAverageTime()) +
                          "\tAvgFPS: " + str(self.getAverageFPS()) +
                          "    ", end="")

                    #result.save(os.path.join('result', img_name))
            print()

            return

        self.resetTimer()
        total_num = run_episode * len(image_file_name_list)
        timer_skipped_num = 0

        for i in range(run_episode):
            for image_file_name in image_file_name_list:
                #  image_file_path = os.path.join(image_folder_path, image_file_name)
                image = jt.random([10, 3, 224, 224])

                self.startTimer()

                result = self.detect(image)

                if timer_skipped_num < timer_skip_num:
                    self.endTimer(False)
                    timer_skipped_num += 1
                else:
                    self.endTimer()

                print("\rNet: ResNet" + str(self.net_depth) +
                      "\tDetected: " + str(self.detected_num) + "/" + str(total_num) +
                      "\t\tAvgTime: " + str(self.getAverageTime()) +
                      "\tAvgFPS: " + str(self.getAverageFPS()) +
                      "    ", end="")

                #result.save(os.path.join('result', img_name))
        print()


if __name__ == '__main__':
    try:
        os.makedirs('result/')
    except:
        print('Destination dir exists')
        pass

    detector_list = []

    for i in range(10):
        detector_list.append(ResNetDetector())
        detector_list[i].loadModel(50)

    for j in range(10):
        for i in range(len(detector_list)):
            detector_list[i].test('sample_images', 100)

