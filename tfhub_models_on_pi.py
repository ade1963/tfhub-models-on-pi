# https://www.tensorflow.org/hub/tutorials/tf2_object_detection

import itertools
import os

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

import time
from six import BytesIO
from six.moves.urllib.request import urlopen
from pathlib import Path
import utils
import tempfile
import requests
import tarfile
import shutil
from pathlib import Path
import csv
from collections import Counter
import datetime

MODE_LOAD_ONLY_SAVED_MODEL = True

MIN_SCORE = 0.1
MAX_OBJECTS = 20

if MODE_LOAD_ONLY_SAVED_MODEL:
    PATH_TO_LABELS = './mscoco_label_map.pbtxt'
else:
    PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
MODEL_METRICS_CSV = './hub_models.csv'
IMAGE_METRICS_CSV = './hub_images.csv'
MODEL_TAR_GZ = os.path.join(tempfile.gettempdir(), 'model.tar.gz')
MODEL_TEMP_FOLDER = os.path.join(tempfile.gettempdir(), 'models')
OUTPUT_IMAGES_PATH = './images'

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

Path(OUTPUT_IMAGES_PATH).mkdir(parents=True, exist_ok=True)


# source: https://www.tensorflow.org/hub/tutorials/tf2_object_detection
ALL_IMAGES = {
  'Beach' : './image2.jpg', # 'models/research/object_detection/test_images/image2.jpg',
  'Dogs' : './image1.jpg', # 'models/research/object_detection/test_images/image1.jpg',
  # By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
  'Naxos Taverna' : 'https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_Coleoptera_of_the_British_islands_(Plate_125)_(8592917784).jpg
  'Beatles' : 'https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg',
  # By Américo Toledano, Source: https://commons.wikimedia.org/wiki/File:Biblioteca_Maim%C3%B3nides,_Campus_Universitario_de_Rabanales_007.jpg
  'Phones' : 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg',
  # Source: https://commons.wikimedia.org/wiki/File:The_smaller_British_birds_(8053836633).jpg
  'Birds' : 'https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg',
}


if MODE_LOAD_ONLY_SAVED_MODEL:
    ALL_IMAGES['Beach'] =  './image2.jpg'
    ALL_IMAGES['Dogs'] = './image1.jpg'


# source: https://www.tensorflow.org/hub/tutorials/tf2_object_detection
ALL_MODELS = {
    'CenterNet HourGlass104 512x512': 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1',
    'CenterNet HourGlass104 Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1',
    'CenterNet HourGlass104 1024x1024': 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024/1',
    'CenterNet HourGlass104 Keypoints 1024x1024': 'https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1',
    'CenterNet Resnet50 V1 FPN 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1',
    'CenterNet Resnet50 V1 FPN Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1',
    'CenterNet Resnet101 V1 FPN 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',
    'CenterNet Resnet50 V2 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512/1',
    'CenterNet Resnet50 V2 Keypoints 512x512': 'https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1',
    'EfficientDet D0 512x512': 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
    'EfficientDet D1 640x640': 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
    'EfficientDet D2 768x768': 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
    'EfficientDet D3 896x896': 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
    'EfficientDet D4 1024x1024': 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
    'EfficientDet D5 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
    'EfficientDet D6 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
    'EfficientDet D7 1536x1536': 'https://tfhub.dev/tensorflow/efficientdet/d7/1',
    'SSD MobileNet v2 320x320': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
    'SSD MobileNet V1 FPN 640x640': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1',
    'SSD MobileNet V2 FPNLite 320x320': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1',
    'SSD MobileNet V2 FPNLite 640x640': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1',
    'SSD ResNet50 V1 FPN 640x640 (RetinaNet50)': 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1',
    'SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)': 'https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_1024x1024/1',
    'SSD ResNet101 V1 FPN 640x640 (RetinaNet101)': 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1',
    'SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)': 'https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1',
    'SSD ResNet152 V1 FPN 640x640 (RetinaNet152)': 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_640x640/1',
    'SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)': 'https://tfhub.dev/tensorflow/retinanet/resnet152_v1_fpn_1024x1024/1',
    'Faster R-CNN ResNet50 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
    'Faster R-CNN ResNet50 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1',
    'Faster R-CNN ResNet50 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1',
    'Faster R-CNN ResNet101 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1',
    'Faster R-CNN ResNet101 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1',
    'Faster R-CNN ResNet101 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1',
    'Faster R-CNN ResNet152 V1 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1',
    'Faster R-CNN ResNet152 V1 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1',
    'Faster R-CNN ResNet152 V1 800x1333': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_800x1333/1',
    'Faster R-CNN Inception ResNet V2 640x640': 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1',
    'Faster R-CNN Inception ResNet V2 1024x1024': 'https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1',
    'Mask R-CNN Inception ResNet V2 1024x1024': 'https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1'
}


def load_images(images):
    results = []
    for img_name, img_url in images.items():
        image_np = utils.load_image_into_numpy_array(img_url)
        results.append({
            "name": f"{img_name}_{image_np.shape[1]}",
            "image_np": image_np,
        })
        image_np = utils.load_image_into_numpy_array(img_url, 640)
        results.append({
            "name": f"{img_name}_{image_np.shape[1]}",
            "image_np": image_np,
        })
    return results


def download_model(model_url):
    start = time.time()
    try:
        shutil.rmtree(MODEL_TEMP_FOLDER)
    except OSError:
        pass
    download_url = model_url.replace('tfhub.dev', 'storage.googleapis.com/tfhub-modules', ) + '.tar.gz'
    r = requests.get(download_url, allow_redirects=True)
    open(MODEL_TAR_GZ, 'wb').write(r.content)
    tar = tarfile.open(MODEL_TAR_GZ, "r:gz")
    tar.extractall(path=MODEL_TEMP_FOLDER)
    tar.close()
    os.remove(MODEL_TAR_GZ)
    print(f'model downloaded in {time.time() - start:.0f} sec')


def eval_image(detector, model_name, image, coco_labels, n_repeats=5):
    start = time.time()
    for _ in range(n_repeats):
        result = detector(image['image_np'])
    tm_avg = (time.time() - start)/n_repeats
 
    result = {key: value.numpy() for key, value in result.items()}

    obj_ids = [id for id, score in zip(result["detection_classes"][0,:MAX_OBJECTS].tolist(), result["detection_scores"][0].tolist()) if score >= MIN_SCORE]
    count_objects = Counter([coco_labels[id] for id in obj_ids])
    count_objects = sorted(count_objects.items(), key=lambda x: x[1], reverse=True)

    image_with_boxes = utils.draw_boxes(
        np.copy(image['image_np'][0, :, :, :]), result["detection_boxes"][0],
        result["detection_classes"][0], result["detection_scores"][0], coco_labels, max_boxes = MAX_OBJECTS, min_score=MIN_SCORE)

    im = Image.fromarray(image_with_boxes)
    im.save(os.path.join(OUTPUT_IMAGES_PATH,
            f'{image["name"]}_{model_name}.png'))

    metrics = {
        'model_name': model_name,
        'image_name': image['name'],
        'objects': str(dict(count_objects)),
        'avg_image': tm_avg,
    }
    return metrics


def eval_model(model_handle, model_name, images):
    image_metrics = []
    start = time.time()
    print(f'start loading model {model_name}')
    detector = hub.load(model_handle)
    tm_load = time.time() - start
    print(f'model {model_name} loaded {tm_load:.0f} sec')
    #     './ssd_hourglass_512',
    # print(detector.summary())
    if not images:
        images = load_images(ALL_IMAGES)
    coco_labels = utils.protobuf_labels(PATH_TO_LABELS)
    if not MODE_LOAD_ONLY_SAVED_MODEL:
        with open(f'{model_name}_labels.json', "w") as fh:
            json.dump(coco_labels, fh, indent=4, sort_keys=True)
    # first prediction to warmup model
    start = time.time()
    detector(images[0]["image_np"])
    tm_warmup = time.time() - start
    print(f'model warmup {tm_warmup:.2f} sec')
    for image in images:
        image_metrics.append(eval_image(detector, model_name, image, coco_labels))
    
    metrics = {
        'name': model_name,
        'init_time': tm_load,
        'warm_up': tm_warmup,
        'graph_info': str(detector.graph_debug_info),
        'tf_git_version': str(detector.tensorflow_git_version),
        'tf_version': str(detector.tensorflow_version),
    }

    return metrics, image_metrics, images


print(datetime.datetime.now())

model_metrics = []
image_metrics = []
images = None
for model_name, model_ref in ALL_MODELS.items():
    model_name = model_name.replace(' ', '_')
    try:
        if MODE_LOAD_ONLY_SAVED_MODEL:
            download_model(model_ref)
            m_metrics, i_metrics, images = eval_model(MODEL_TEMP_FOLDER, model_name, images)
        else:
            try:
                os.remove('./models')
            except:
                pass  
            m_metrics, i_metrics, images = eval_model(model_ref, model_name, images)
        model_metrics.append(m_metrics)
        image_metrics.extend(i_metrics)
    except Exception as e:
        print(f'Exception in model {model_name}')
        print(e)

with open(MODEL_METRICS_CSV, 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = list(model_metrics[0].keys()))
    writer.writeheader() 
    writer.writerows(model_metrics) 

with open(IMAGE_METRICS_CSV, 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = list(image_metrics[0].keys()))
    writer.writeheader() 
    writer.writerows(image_metrics) 

# clear temp files
try:
    os.remove(MODEL_TAR_GZ)
    os.remove(MODEL_TEMP_FOLDER)
except:
    pass

print(datetime.datetime.now())

