from __future__ import print_function
import os

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
import torch
from model import VSE, order_sim
from collections import OrderedDict
import pickle
import itertools

from vocab import Vocabulary
import evaluation

split = "val"
modelpath = "runs/coco_vse++_resnet_restval_finetune/model_best.pth.tar"
datapath = "data/"
# evaluation.evalrank(modelpath, data_path=datapath, split="val")

# load model and options
if torch.cuda.is_available():
    checkpoint = torch.load(modelpath)
else:
    checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)

opt = checkpoint['opt']
opt.data_path = datapath

with open(os.path.join(opt.vocab_path,'%s_vocab.pkl' % opt.data_name),'rb') as f:
    vocab = pickle.load(f)

opt.vocab_size = len(vocab)

# construct model
model = VSE(opt)
# load model state
model.load_state_dict(checkpoint['model'])

print('Loading dataset')
data_loader = get_test_loader(split, opt.data_name,
                              vocab, opt.crop_size,
                              opt.batch_size, opt.workers,
                              opt)

# switch to evaluate mode
model.val_start()

# numpy array to keep all the embeddings
img_embs = None
cap_embs = None

# numpy array accum
accumvecs = None

idslist = []
pathslist = []

for i, (images, _, lengths, ids, paths) in enumerate(data_loader):
    print("Computing batch-" + str(i))
    if not torch.cuda.is_available():
        model.img_enc = model.img_enc.cpu()
        model.txt_enc = model.txt_enc.cpu()

    imgvec = model.img_enc(torch.autograd.Variable(images, volatile=True))

    idslist.append(ids)
    pathslist.append(paths)

    if accumvecs is None:
        accumvecs = np.array(imgvec.data.cpu().numpy())
    else:
        accumvecs = np.append(accumvecs, imgvec.data.cpu().numpy(), axis=0)

    print(accumvecs.shape)

all_ids = [i for i in itertools.chain(*idslist)]
all_paths = [i for i in itertools.chain(*pathslist)]

pickle.dump(accumvecs, open("val2014imgvecs.pkl", "wb"))
pickle.dump(all_ids, open("val2014ids.pkl", "wb"))
pickle.dump(all_paths, open("val2014paths.pkl", "wb"))