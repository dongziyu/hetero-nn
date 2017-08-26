#!/usr/bin/env python
'''This script extracts pretrained caffe model based on net prototype and writes
it as a C++ includeable header file. Caffe model file and net file should match.
This work is not necessary for most of applications, only aims at preparing CNN
model as constants in Halide.
'''

# pylint: disable=invalid-name, missing-docstring

import caffe

WEIGHT = 0
BIAS = 1

def extract_caffemodel(proto, model):

    with open('params.h', 'w') as f:
        f.write('#ifndef __PARAMS_H__\n')
        f.write('#define __PARAMS_H__\n\n')

        net = caffe.Net(proto, model, caffe.TEST)
        for layer, blob in net.params.iteritems():
            f.write('const double %s_w' % layer)
            shape = blob[WEIGHT].data.shape
            # for conv layers
            if len(shape) == 4:
                i, j, k, m = shape
                f.write('[][%s][%s][%s] = ' % (j, k, m))
                f.write('{')
                for ii in xrange(i):
                    f.write('{')
                    for jj in xrange(j):
                        f.write('{')
                        for kk in xrange(k):
                            f.write('{')
                            f.write(', '.join(
                                str(blob[WEIGHT].data[ii][jj][kk][mm])
                                for mm in xrange(m)))
                            f.write('}, ')
                        f.write('}, ')
                    f.write('}, ')
                f.write('};\n\n')
            # for ip layers
            if len(shape) == 2:
                i, j = shape
                f.write('[][%s] = ' % j)
                f.write('{')
                for ii in xrange(i):
                    f.write('{')
                    f.write(', '.join(
                        str(blob[WEIGHT].data[ii][jj]) for jj in xrange(j)))
                    f.write('}, ')
                f.write('};\n\n')

            length = blob[BIAS].shape[0]
            f.write('const double %s_b[%s] = ' % (layer, length))
            f.write('{')
            f.write(', '.join(str(blob[BIAS].data[i]) for i in xrange(length)))
            f.write('};\n\n')

        f.write('#endif /* __PARAMS_H__ */\n')

if __name__ == '__main__':
    extract_caffemodel('lenet.prototxt', 'lenet_iter_10000.caffemodel')
