from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate
'''
class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train)
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = dataloader.DataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

 

        if args.data_test in ['DenoiseSet68','DenoiseColorSet68','DenoiseGrayDIV2K','DenoiseColorDIV2K']:

            module_test = import_module('data.benchmarkdenoise')
            testset = getattr(module_test, 'BenchmarkDenoise')(args, train=False)
        elif args.data_test in ['ISPfuji_test','ISPsony_test','ISPfuji_val','ISPsony_val']:
            module_test = import_module('data.benchmarkisp')
            testset = getattr(module_test, 'BenchmarkISP')(args, train=False)
        else:
            module_test = import_module('data.' +  args.data_test)
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = dataloader.DataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
'''


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train)
            print('进入该模型：', module_train)
            trainset = getattr(module_train, args.data_train)(args)
            '''
            数据导入流程：
            File"G:/MyPythonCode/SGN_ICCV2019/main.py", line15, in < module >
              loader = data.Data(args)
            File"G:\MyPythonCode\SGN_ICCV2019\data\__init__.py", line 72, in __init__
              testset = getattr(module_test, 'BenchmarkDenoise')(args, train=False, name=args.data_test)
            File"G:\MyPythonCode\SGN_ICCV2019\data\benchmarkdenoise.py", line14, in __init__
              super(BenchmarkDenoise, self).__init__(args, name=name, train=train, benchmark=True)
            File"G:\MyPythonCode\SGN_ICCV2019\data\ImageData.py", line35, in __init__
              _load_benchmark_bin()
            File"G:\MyPythonCode\SGN_ICCV2019\data\ImageData.py", line24, in _load_benchmark_bin
              self.images_tar = np.load(self._name_tarbin())
            File"D:\software\Anaconda3\envs\pytorch\lib\site-packages\numpy\lib\npyio.py", line 428, in load
                fid = open(os_fspath(file), "rb")
            FileNotFoundError: [Errno 2]
                No such file or directory: 'G:/Datasets/SGN/BenchmarkDenoise\\DenoiseSet68\\bin\\test_bin_tar.npy'
            '''
            # args.data_train---DIV2KDENOISE
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []

        if args.data_test in ['DenoiseSet68','DenoiseColorSet68','DenoiseGrayDIV2K','DenoiseColorDIV2K']:

            module_test = import_module('data.benchmarkdenoise')
            testset = getattr(module_test, 'BenchmarkDenoise')(args, train=False, name = args.data_test)
        elif args.data_test in ['ISPfuji_test','ISPsony_test','ISPfuji_val','ISPsony_val','ISPfuji_val20','ISPsony_val20']:
            module_test = import_module('data.benchmarkisp')
            testset = getattr(module_test, 'BenchmarkISP')(args, train=False, name = args.data_test)
        else:
            module_test = import_module('data.' + args.data_test)
            testset = getattr(module_test, args.data_test)(args, train=False, name = args.data_test)


 

        self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
