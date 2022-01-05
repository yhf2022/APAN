import os
import numpy as np
import colorsys
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
from utils.utils import letterbox_image, non_max_suppression, yolo_correct_boxes
from torchsummary import summary
from nets.yolo4 import APAN
from map import map_get
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='logs\Epoch102-Total_Loss10.9053-Val_Loss15.2265.pth', help="weights", action="store_true")
parser.add_argument('--classes_path',default='model_data/classes.txt', help='the classes of targets', action="store_true")
parser.add_argument('--_summary',default=False, help="the framework of model", action="store_true")
parser.add_argument('--_test_id',default='VOCdevkit/VOC2007/ImageSets/Main/test.txt', help="the address of testing", action="store_true")
args = parser.parse_args()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class mAP_Yolo_APAN():

    def __init__(self,model_path,classes_path, **kwargs):
      
        self.model_path = model_path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = classes_path
        self.model_image_size = (416, 416, 3)
        self.confidence = 0.05
        self.iou = 0.5
        self.cuda = True
        self.letterbox_image = False
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    def generate(self):

        self.net = APAN(len(self.anchors[0]), len(self.class_names)).eval()


        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
        self.yolo_decodes.append(DecodeBox(self.anchors[3], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self,image_id,image):
        self.confidence = 0.05
        self.iou = 0.5
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])

        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))

        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            output_list = []
            for i in range(4):
                output_list.append(self.yolo_decodes[i](outputs[i]))


            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return 
  
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)


            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

if __name__ == "__main__":

    yolo = mAP_Yolo_APAN(args.model_path,args.classes_path)
    image_ids = open(args._test_id).read().strip().split()

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")


    for image_id in tqdm(image_ids):
        image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
        image = Image.open(image_path)
        yolo.detect_image(image_id,image)

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/ground-truth"):
        os.makedirs("./input/ground-truth")

    for image_id in image_ids:
        with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
            root = ET.parse("VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text

                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = APAN(3,20).to(device)
    if args._summary:
        summary(model, input_size=(3, 416, 416))
    print("Conversion completed!")
    map_get()
    

    


