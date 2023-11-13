from lib import *
import urllib.request
import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='./data', help='path to directory to stored data')
arg = parser.parse_args()
data_dir = arg.path
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url_img = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
url_annotations = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
target_path = os.path.join(data_dir, 'Flickr8k')
anno_path = os.path.join(data_dir, 'Annotations')
if not os.path.exists(target_path):
    urllib.request.urlretrieve(url_img, target_path)

    zip = zipfile.ZipFile(target_path)

    zip.extractall(data_dir)
    zip.close

if not os.path.exists(anno_path):
    urllib.request.urlretrieve(url_annotations, anno_path)

    zip = zipfile.ZipFile(anno_path)

    zip.extractall(data_dir)
    zip.close