# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import argparse
import numpy as np


def main():
    """Generate fake numpy binary dataset

    numpy_dataset.py --fname='' --nlabels=10 --shape=784 --size=1000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=False, default='/dev/shm/data/dataset.npz', help='Output file name.')
    parser.add_argument('--nlabels', type=int, required=False, default=10, help='Number of labels.')
    parser.add_argument('--size', type=int, required=False, default=1024*100, help='Number of instances in dataset.')
    parser.add_argument('--shape', type=str, required=False, default='784', help='Shape of one instance.')
    args = parser.parse_args()
    if not args.fname.endswith('.npz'):
        raise ValueError("Wrong output  file name. Extension must be '.npz'.")
    if args.nlabels is not None and args.nlabels <= 0:
        raise ValueError("Number of labels must be a positive number.")
    if args.size <= 0:
        raise ValueError("Dataset size must be a positive number.")

    data_shape = [int(dim) for dim in args.shape.split(',')]
    for dim in data_shape:
        if dim <= 0:
            raise ValueError("Shape dimension must be a positive integer number.")

    data = np.random.uniform(low=-1, high=1, size=(args.size,)+tuple(data_shape))
    if args.nlabels:
        labels = np.random.randint(low=0, high=args.nlabels, size=(args.size, 1), dtype='int')
    else:
        labels = None
    print ("Data shape: %s, labels shape: %s" % (data.shape, labels.shape))

    np.savez(args.fname, data=data, labels=labels)


def test_mxnet_iterator(fname='dataset.npz'):
    import mxnet as mx

    dataset = np.load(fname)
    if 'data' not in dataset:
        raise "Dataset does not provide data."
    labels = dataset['labels'] if 'labels' in dataset else None
    if labels is None:
        print ("[WARNING] no labels found, assuming unsupervised training.")
    dataiter = mx.io.NDArrayIter(data=dataset['data'], label=labels,
                                 batch_size=50, shuffle=True,
                                 last_batch_handle='discard')
    for batch in dataiter:
        print ("Batch: {data_shape=%s, labels_shape=%s}" % (batch.data[0].shape, batch.label[0].shape))


if __name__ == '__main__':
    main()
    #test_mxnet_iterator();
