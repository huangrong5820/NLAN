import os
from data import common
from data import ImageData

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class BenchmarkDenoise(ImageData.ImageData):
    def __init__(self, args, name = '', train=True):
        super(BenchmarkDenoise, self).__init__(args, name = name, train=train, benchmark=True)
        self.ext = '.png'
        def _load_benchmark_bin():
            self.images_tar = np.load(self._name_tarbin())
            self.images_input = np.load(self._name_inputbin())

        def _load_bin():
            self.images_tar = np.load(self._name_tarbin())
            self.images_input = np.load(self._name_inputbin())

        print('initial image data now!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('benchmark', self.benchmark)

        if self.benchmark and self.testbin:
            # args.testbin----benchmark useing bin data
            print('LOAD BENCHMARK BIN')
            _load_benchmark_bin()
            print('BIN LOAD SUCCESSED!')
        elif self.benchmark:
            print('BenchmarkScaning')
            self.images_tar, self.images_input = self._scan()
            #print(self.images_tar)
            print('Scan finished!')
        elif args.ext == 'img':
            self.images_tar, self.images_input = self._scan()
        elif args.ext.find('sep') >= 0:

            print('TrainingDataScaning')
            self.images_tar, self.images_input = self._scan()
            print('Scan finished!')
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_tar:
                    img_tar = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_tar)
                for v in self.images_input:
                    img_input = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, img_input)
            if(self.ext=='.png'):
                self.images_tar = [v.replace(self.ext, '.npy') for v in self.images_tar            ]
                self.images_input = [v.replace(self.ext, '.npy') for v in self.images_input]

        elif args.ext.find('bin') >= 0:
            print('bibbiibibibibibibibibibbibibibibibibibibi')
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                print('self.apath', self.apath)            # G:/Datasets/SGN//DIV2K
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_tar, list_input = self._scan()
                img_tar = [misc.imread(f) for f in list_tar]
                np.save(self._name_tarbin(), img_tar)
                del img_tar

                img_input = [misc.imread(f) for f in list_input]
                np.save(self._name_inputbin(), img_input)
                del img_input

                _load_bin()
        elif args.ext.find('memmap') >= 0:
            print(self._sample_number(),self._height_input(),self._width_input(),self._c_in())
            a = np.memmap(self._name_inputmap(),dtype='float32',mode='r',shape=(self._sample_number(),self._height_input(),self._width_input(),self._c_in()))
            self.images_tar = np.memmap(self._name_targetmap(),dtype='float32',mode='r',shape=(self._sample_number,self._height_target(),self._width_target(),self._c_out()))

        else:
            print('Please define data type')

    def _scan(self):
        list_tar = []
        list_input = []
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_tar.append(os.path.join(self.dir_tar, filename + self.ext))
            list_input.append(os.path.join(self.dir_input,filename + self.ext))

        return list_tar, list_input


    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'BenchmarkDenoise', self.args.data_test)
        self.dir_tar = self.apath
        self.dir_input = self.apath

    def _name_tarbin(self):
        return os.path.join(
            self.apath,
            '{}_bin_tar.npy'.format(self.split)
        )

    def _name_inputbin(self):
        return os.path.join(
            self.apath,
            '{}_bin_x{}_sigma{}.npy'.format(self.split, self.scale, self.sigma)
        )
