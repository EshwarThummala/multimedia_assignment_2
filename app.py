from flask import Flask, render_template, url_for,request,redirect
import json
import numpy as np
from scipy import spatial
import cv2
from flask_bootstrap import Bootstrap
import util

# Create a Flask web application instance.
app = Flask(__name__)
Bootstrap(app)

# Define a route for the root URL ('/') with support for GET and POST methods.
@app.route('/', methods =['GET','POST'] )
def index():
    print(request.method)
    if(request.method == 'POST'):
      dictk = dict()
      # Check if the provided image_id is within a valid range.
      if(-1 < int(request.form['image_id']) < 8677):
         dictk = get_descriptors(request.form['image_id'], int(request.form['k']), request.form['vectorspace'], request.form['latentspace'])
         # Add the current image's path to the result dictionary.
         dictk['cur_image'] = 'static/torchvision_images/'+request.form['image_id']+'.jpg'
      else:
         # Handle the case where the provided image is not found in the dataset.
         dictk['image_not_found'] = 'The provided image is is not found in the dataset'
      result = json.loads(json.dumps(dictk))
      return render_template('index.html', result = result)
    return render_template('index.html', result=None)

# Different distance measure for the distance calculations
def cubrt(cur_desc, check_desc):
  return np.cbrt(np.sum(np.power(abs(cur_desc - check_desc) , 3)))

def euc(cur_desc, check_desc):
  return np.sqrt(np.sum(np.power(abs(cur_desc - check_desc) , 2)))

def linea(cur_desc, check_desc):
  return np.sum(abs(cur_desc - check_desc))

def cosinek(cur_desc, check_desc):
  return spatial.distance.cosine(cur_desc,  check_desc)

# Abstaract image to get k images based on the vector space
def get_descriptors(image_id, k, vectorspace, latentspace):
   mappings = dict()
   if(latentspace == 'none'):
      return get_vectorspace_matches(image_id, k, vectorspace)
      '''if(vectorspace == 'color'):
         return get_color_desc_matches(image_id, k)
      elif(vectorspace == 'hog'):
         return get_hog_desc_matches(image_id, k)
      elif(vectorspace == 'layer3'):
         return get_layer3_desc_matches(image_id, k)
      elif(vectorspace == 'avgpool'):
         return get_avgpool_desc_matches(image_id, k)
      elif(vectorspace == 'fc'):
         return get_fc_desc_matches(image_id, k)'''
   else:
      return get_latentspace_matches(image_id, k, vectorspace, latentspace)

def top_k_labels(distances, k):
   avg_labels = dict()
   label_count = dict()
   for i in distances:
      if(i[2] in avg_labels):
         avg_labels[i[2]] += i[0]
         label_count[i[2]] += 1
      else:
         avg_labels[i[2]] = i[0]
         label_count[i[2]] = 1
   for i in avg_labels.keys():
      avg_labels[i] = avg_labels[i]/label_count[i]
   return list(sorted(avg_labels.items(), key=lambda item: item[1]))[:k]

# readle format of the descriptors for respective vector spaces
'''def get_color_display_descriptor(descr):
  res = []
  k = 0
  for color in ['Red_channel', 'Green_channel', 'Blue_channel']:
      for i in range(1,101):
         for j in range(3):
            metrics = ['Mean','STD','Skew']
            res.append(color+'/partition-'+str(i)+'/'+metrics[j%3]+' = '+str(descr[k]))
            k += 1
  return res

def get_hog_display_descriptor(descr):
  res = []
  angles = ['0-40','41-80','81-120','121-160','161-200','201-240','241-280','281-320','321-360']
  k = 0
  for i in range(100):
     for j in range(9):
        res.append('Partition-'+str(i+1)+'/'+angles[j]+' bin = '+str(descr[k]))
        k += 1
  return res

def get_resnet_diaplsy_descriptor(descr):
   res = []
   for i in range(1,len(descr)+1):
      res.append('FD value-'+str(i)+' = '+str(descr[i-1]))
   return res

# Color vector space image feature descriptor generation functions
def get_color_desc_matches(image_id, k):
    json_file = open('descriptors\cd_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fc_dict = json.loads(loaded_model_json)
    if(image_id not in fc_dict.keys()):
       fictk = dict()
       fictk['error'] = 'Given image is gray scale and it only has one channel'
       return fictk
    cur_desc = np.asarray(fc_dict[image_id]['feature-descriptor'])
    res_dict = dict()
    res_dict['cur_desc'] = get_color_display_descriptor(cur_desc)
    distances = []
    for key in fc_dict.keys():
        check_desc = np.asarray(fc_dict[key]['feature-descriptor'])
        dist =  euc(cur_desc, check_desc)
        distances.append([dist, key])
    distances.sort(key = lambda x: x[0])
    for li in distances[:k]:
       res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
       res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
    return res_dict

# HOG vector space image feature descriptor generation functions
def get_hog_desc_matches(image_id, k):
    json_file = open('descriptors\hog_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fc_dict = json.loads(loaded_model_json)
    cur_desc = np.asarray(fc_dict[image_id]['feature-descriptor'])
    res_dict = dict()
    res_dict['cur_desc'] = get_hog_display_descriptor(cur_desc)
    distances = []
    for key in fc_dict.keys():
        check_desc = np.asarray(fc_dict[key]['feature-descriptor'])
        dist =  euc(cur_desc, check_desc)
        distances.append([dist, key])
    distances.sort(key = lambda x: x[0])
    i = 1
    for li in distances[:k]:
       res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
       res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
    return res_dict

# Layer3 vector space image feature descriptor generation functions
def get_layer3_desc_matches(image_id, k):
    json_file = open('descriptors\layer3_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fc_dict = json.loads(loaded_model_json)
    if(image_id not in fc_dict.keys()):
       fictk = dict()
       fictk['error'] = 'Either the image is gray scale or the image is not present in the dataset please give a proper input'
       return fictk
    cur_desc = np.asarray(fc_dict[image_id]['feature-descriptor'])
    res_dict = dict()
    res_dict['cur_desc'] = get_resnet_diaplsy_descriptor(cur_desc.tolist())
    distances = []
    for key in fc_dict.keys():
        check_desc = np.asarray(fc_dict[key]['feature-descriptor'])
        dist =  euc(cur_desc, check_desc)
        distances.append([dist, key])
    distances.sort(key = lambda x: x[0])
    for li in distances[:k]:
       res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
       res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
    return res_dict

# Avgpool vector space image feature descriptor generation functions
def get_avgpool_desc_matches(image_id, k):
    json_file = open('descriptors\avgpool_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fc_dict = json.loads(loaded_model_json)
    image_id = image_id.split('.')[0]
    if(image_id.split('.')[0] not in fc_dict.keys()):
       fictk = dict()
       fictk['error'] = 'Either the image is gray scale or the image is not present in the dataset please give a proper input'
       return fictk
    cur_desc = np.asarray(fc_dict[image_id]['feature-descriptor'])
    res_dict = dict()
    res_dict['cur_desc'] = get_resnet_diaplsy_descriptor(cur_desc.tolist())
    distances = []
    for key in fc_dict.keys():
        check_desc = np.asarray(fc_dict[key]['feature-descriptor'])
        dist =  euc(cur_desc, check_desc)
        distances.append([dist, key])
    distances.sort(key = lambda x: x[0])
    k_labels = top_k_labels(distances, k)
    res_dict['labels'] = k_labels    
    for li in distances[:k]:
       res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
       res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
    return res_dict

# Fully connected layer vector space image feature descriptor generation functions
def get_fc_desc_matches(image_id, k):
    json_file = open('descriptors\fc_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    fc_dict = json.loads(loaded_model_json)
    if(image_id not in fc_dict.keys()):
       fictk = dict()
       fictk['error'] = 'Either the image is gray scale or the image is not present in the dataset please give a proper input'
       return fictk
    cur_desc = np.asarray(fc_dict[image_id]['feature-descriptor'])
    res_dict = dict()
    res_dict['cur_desc'] = get_resnet_diaplsy_descriptor(cur_desc.tolist())
    distances = []
    for key in fc_dict.keys():
        check_desc = np.asarray(fc_dict[key]['feature-descriptor'])
        dist =  euc(cur_desc, check_desc)
        distances.append([dist, key])
    distances.sort(key = lambda x: x[0])
    k_labels = top_k_labels(distances, k)
    res_dict['labels'] = k_labels
    for li in distances[:k]:
       res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
       res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
    return res_dict
'''
def get_vectorspace_matches(image_id, k, vectorspace):
   json_file = open('descriptors\\'+vectorspace+'_desc_a2.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   dictk = json.loads(loaded_model_json)
   if(image_id not in dictk.keys()):
      fictk = dict()
      fictk['error'] = 'Either the image is gray scale or the image is not present in the dataset please give a proper input'
      return fictk
   cur_desc = np.asarray(dictk[image_id]['feature-descriptor'])
   res_dict = dict()
   #res_dict['cur_desc'] = get_resnet_diaplsy_descriptor(cur_desc.tolist())
   distances = []
   for key in dictk.keys():
      check_desc = np.asarray(dictk[key]['feature-descriptor'])
      dist =  euc(cur_desc, check_desc)
      distances.append([dist, key, dictk[key]['label']])
   distances.sort(key = lambda x: x[0])
   k_labels = top_k_labels(distances, k)
   res_dict['labels'] = k_labels
   for li in distances[:k]:
      res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
      res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
   return res_dict   

def get_latentspace_matches(image_id, k, vectorspace,latentspace):
   if(latentspace in ['cpd']):
      json_file = open('descriptors\\latent_descriptors\\'+latentspace+'\\'+vectorspace+'_'+latentspace+'.json', 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      dictk = json.loads(loaded_model_json)
   else:
      dictk = util.get_latent_semantics(k, vectorspace, latentspace)
   if(image_id not in dictk.keys()):
      fictk = dict()
      fictk['error'] = 'Either the image is gray scale or the image is not present in the dataset please give a proper input'
      return fictk
   cur_desc = np.asarray(dictk[image_id]['feature-descriptor'])
   res_dict = dict()
   #res_dict['cur_desc'] = get_resnet_diaplsy_descriptor(cur_desc.tolist())
   distances = []
   for key in dictk.keys():
      check_desc = np.asarray(dictk[key]['feature-descriptor'])
      dist =  euc(cur_desc, check_desc)
      distances.append([dist, key, dictk[key]['label']])
   distances.sort(key = lambda x: x[0])
   k_labels = top_k_labels(distances, k)
   res_dict['labels'] = k_labels
   for li in distances[:k]:
      res_dict[li[1]] = 'static/torchvision_images/'+li[1]+'.jpg'
      res_dict[li[1]+'-score'] = str(float("{:.4f}".format(li[0])))
   return res_dict

if __name__ == "__main__":
    app.run(debug=True)