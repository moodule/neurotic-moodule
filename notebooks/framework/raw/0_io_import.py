"""Functions for importing and exporting the data.
	Downloading / sending
	Compressing / extracting
	Serialization / deserialization"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

#TODO 

def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def load_folder(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

# ==============================================================================================
# CAFFE ========================================================================================
#!/usr/bin/env python

import os
import sys
import numpy as np
import tensorflow as tf

class CaffeLoader(object):
    def __init__(self, params_path):
        self.params = np.load(params_path).item()

    def load(self, sesh):
        for key in self.params:
            print "scope:", key
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), self.params[key]):
                    print subkey
                    print data.shape
                    sesh.run(tf.get_variable(subkey, trainable=False).assign(data))

class CaffeDataReader(object):
    def __init__(self, def_path, data_path):
        self.def_path = def_path
        self.data_path = data_path
        self.load()

    def load(self):
        try:            
            self.load_using_caffe()
        except ImportError:
            print('WARNING: PyCaffe not found!')
            print('Falling back to protocol buffer implementation.')
            print('This may take a couple of minutes.')            
            self.load_using_pb()

    def load_using_caffe(self):
        import caffe
        net = caffe.Net(self.def_path, self.data_path, caffe.TEST)
        data = lambda blob: blob.data
        self.parameters = [(k, map(data, v)) for k,v in net.params.items()]        

    def load_using_pb(self):
        import caffepb
        data = caffepb.NetParameter()
        data.MergeFromString(open(self.data_path, 'rb').read())
        pair = lambda layer: (layer.name, self.transform_data(layer))
        layers = data.layers or data.layer
        self.parameters = [pair(layer) for layer in layers if layer.blobs]

    def transform_data(self, layer):
        transformed = []
        for idx, blob in enumerate(layer.blobs):
            if blob.num and blob.channels and blob.height and blob.width:
                c_o  = blob.num
                c_i  = blob.channels
                h    = blob.height
                w    = blob.width
                data = np.squeeze(np.array(blob.data).reshape(c_o, c_i, h, w))
                transformed.append(data)
            elif len(blob.shape.dim) >= 3:
                c_o  = blob.shape.dim[0]
                c_i  = blob.shape.dim[1]
                h    = blob.shape.dim[2]
                w    = blob.shape.dim[3]
                data = np.squeeze(np.array(blob.data).reshape(c_o, c_i, h, w))
                transformed.append(data)
        return tuple(transformed)

    def dump(self, dst_path):
        params = []
        def convert(data):
            if data.ndim==4:
                # (c_o, c_i, h, w) -> (h, w, c_i, c_o)
                data = data.transpose((2, 3, 1, 0))
            elif data.ndim==2:
                # (c_o, c_i) -> (c_i, c_o)
                data = data.transpose((1, 0))
            # Kludge Alert: The FC layer wired up to the pool layer
            # needs to be re-ordered.
            # TODO: Figure out a more elegant way to do this.        
            if params and params[-1][1][0].ndim==4 and data.ndim==2:
                prev_c_o = params[-1][1][0].shape[-1]
                cur_c_i, cur_c_o = data.shape
                dim = np.sqrt(cur_c_i/prev_c_o)
                data = data.reshape((prev_c_o, dim, dim, cur_c_o))
                data = data.transpose((1, 2, 0, 3))
                data = data.reshape((prev_c_o*dim*dim, cur_c_o))
            return data
        for key, data_pair in self.parameters:
            params.append((key, map(convert, data_pair)))
        print('Saving to %s'%dst_path)
        np.save(open(dst_path, 'wb'), dict(params))
        print('Done.')

