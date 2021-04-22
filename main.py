import torch
import time
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    utility.print_network(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    first_enter = True
    while not t.terminate(first_enter):
        start_epo_t = time.time()
        print("Training!!!")
        t.train()
        print("Evaluating!!!")
        t.test()
        end_epo_t = time.time()
        duration = end_epo_t - start_epo_t
        if first_enter:
            print('Each epoch consumes：{}minutes'.format(duration/60))
            print('This work will end in {} days'.format((duration/3600) * args.epochs / 24))
        first_enter = False
    checkpoint.done()

