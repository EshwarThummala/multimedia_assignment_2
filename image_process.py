import numpy as np
import cv2
from PIL import Image
import app
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import util

resnet_model = models.resnet50(pretrained=True)

''' Support function to get skew for the data passed '''
def getSkew(vector):
  mean = np.mean(vector)
  v = vector - mean
  v = v ** 3
  s = np.sum(v)
  s = s/len(v)
  if(s < 0):
    return 0
  return s ** (1/3)

''' Function to get the color descriptor for the image passed to it'''
def get_color_descriptor(img):

  ''' resizing the image to 300X100 and extracting the color channels from the image '''
  resized_img = img.resize((300,100))
  img_array = np.asarray(resized_img)
  img_array = img_array/255
  red_values = img_array[:,:,0]
  blue_values = img_array[:,:,1]
  green_values = img_array[:,:,2]
  color_descriptor = []

  ''' partitioning the image into 10*10 grid and calculating the Mean, STD and Skew parameters for the partitions '''
  for channel in [red_values, blue_values, green_values]:
    i = 0
    while(i < 100):
      j = 0
      while(j < 300):
        parti = channel[i:i+10,j:j+30]
        mean, std, skewc = np.mean(parti), np.std(parti), getSkew(parti.flatten()) #skew(parti.flatten())
        color_descriptor.append(mean)
        color_descriptor.append(std)
        if(np.isnan(skewc)):
          color_descriptor.append(0)
        else:
          color_descriptor.append(skewc)
        j += 30
      i += 10
  return color_descriptor

''' Function to get the HOG descriptor for the image passed to it '''
def get_hog_descriptor(hog_img):
  ''' resizing the image to 300X100 and generating gx, gy, magnitude and angle of the images using cv2 package '''
  hog_img = hog_img.resize((300,100))
  img_arry = np.asarray(hog_img)
  gx = cv2.Sobel(img_arry, cv2.CV_32F, 1, 0, ksize=1)
  gy = cv2.Sobel(img_arry, cv2.CV_32F, 0, 1, ksize=1)
  mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

  ''' Dividing the mag abd angle into partitions and generating 9 bins comprising the equal angle intervals of 40 from 0 to 360 '''
  bin_width = 360.0 / 9
  hog_descriptor = []
  i = 0
  while(i < 100):
    j = 0
    while(j < 300):
      partm = mag[i:i+10,j:j+30]
      parta = angle[i:i+10,j:j+30]
      hist = np.zeros((9,))
      for k in range(9):
        bin_start = k * bin_width
        bin_end = (k + 1) * bin_width
        mask = (parta >= bin_start) & (parta < bin_end)
        hist[k] = np.sum(partm[mask])
      hog_descriptor += hist.tolist()
      j += 30
    i += 10
  return hog_descriptor

def get_layer3_descriptor(img):
  def hook_layer3(module,input,output):
    global fd_layer3
    fd_layer3 = output
  def process_layer3(descr):
    res = []
    for i in range(1024):
        res.append(np.mean(descr[i]).tolist())
    return res
  preprocess = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
  ])
  img_tensor = preprocess(img)
  img_tensor = img_tensor.unsqueeze(0)
  layer3_layer = resnet_model.layer3
  layer3_hook = layer3_layer.register_forward_hook(hook_layer3)
  with torch.no_grad():
    fc_desc = resnet_model(img_tensor).numpy().flatten().tolist()
  return process_layer3(fd_layer3.numpy()[0])

def get_avgpool_descriptor(img):
  def hook_avgpool(module,input,output):
    global fd_avg_pool
    fd_avg_pool = output
  def process_avgpool(descr):
    res = []
    i = 0
    while(i < 2048):
        res.append((descr[i]+descr[i+1])/2)
        i +=2
    return res
  preprocess = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
  ])
  img_tensor = preprocess(img)
  img_tensor = img_tensor.unsqueeze(0)
  avgpool_layer = resnet_model.avgpool
  avgpool_hook = avgpool_layer.register_forward_hook(hook_avgpool)
  with torch.no_grad():
    fc_desc = resnet_model(img_tensor).numpy().flatten().tolist()
  return process_avgpool(fd_avg_pool.flatten().tolist())

def get_fc_descriptor(img):
  preprocess = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
  ])
  img_tensor = preprocess(img)
  img_tensor = img_tensor.unsqueeze(0)
  with torch.no_grad():
    fc_desc = resnet_model(img_tensor).numpy().flatten().tolist()
  return fc_desc
  
def get_newimage_matches(imagePath, vectorspace, latentspace, k):
  image = Image.open(imagePath)
  curr_desc = []
  if(image.mode == "RGB"):
    if(vectorspace == 'color'):
        curr_desc = get_color_descriptor(image)
    elif(vectorspace == 'hog'):
        curr_desc = get_hog_descriptor(image)
    elif(vectorspace == 'layer3'):
        curr_desc = get_layer3_descriptor(image)
    elif(vectorspace == 'avgpool'):
        curr_desc = get_avgpool_descriptor(image)
    elif(vectorspace == 'fc'):
        curr_desc = get_fc_descriptor(image)
    if(latentspace == 'none'):
        return app.get_vectorspace_matches(curr_desc, k, vectorspace, True)
    latent_feature = util.get_objectlatent_matrix([curr_desc], k, latentspace).flatten().tolist()
    return app.get_latentspace_matches(latent_feature, k, vectorspace, latentspace, True)
  else:
     dictk = {}
     dictk['error'] = 'Given image is gray scale'