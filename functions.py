import logging
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import cv2
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)
    
    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)
    
    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    
    return my_logger

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊

def showpic(pic):              #顯示圖片
    cv2.imshow('RGB', pic)     #顯示 RGB 的圖片
    cv2.waitKey(0)             #有這段才不會有bug

def readpic(p):                #讀入圖片
    return cv2.imread(p)
    
def savepic(img, p):           #儲存圖片
    cv2.imwrite(p, img)
    
# 處理多張圖片變成 tensor
def pic2tensor(img_list):
    pic = []
    n = img_list.shape[0]
    for i in range(n):
        path = img_list[i]
        image = readpic(path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        t = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        # t = (t - 0.5) / 0.5
        t = t.view(1, 3, 224, 224)
        pic.append(t)
    outputs = torch.cat(pic, dim=0)
    return outputs

def pic2tensor_2(img_list):
    pic = []
    n = img_list.shape[0]
    for i in range(n):
        path = img_list[i]
        image = readpic(path)
        t = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        # t = (t - 0.5) / 0.5
        t = t.view(1, 3, 512, 512)
        pic.append(t)
    outputs = torch.cat(pic, dim=0)
    return outputs

def sumlist(l, n):
    c = 0
    for i in l:
        c += i
    return c / n

def getf1(moduleOutputs, reals):
    pred = torch.argmax(moduleOutputs, dim=1)
    c = f1_score(reals, pred, average='macro')
    return c

def getaccu(moduleOutputs, reals):
    pred = torch.argmax(moduleOutputs, dim=1)
    c = accuracy_score(reals, pred)
    return c

def train(modules, inputs, labels, update):
    module = modules[0]
    lossf = modules[1]
    opt = modules[2]
    
    if update:
        outputs = module(inputs)
        loss = lossf(outputs, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item()
    
    else:
        n = inputs.shape[0]
        index = 0
        with torch.no_grad():
            outputs = []
            while index < n:
                out = module(inputs[index:index+5, :, :, :])
                outputs.append(out)
                index += 5
            outputs = torch.cat(outputs, dim=0)
        return outputs

def tryf():
    print('00000')