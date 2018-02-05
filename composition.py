import os
import pickle
import argparse
import torch
import numpy as np

from compositionutils import im_utils
from compositionutils import data_utils

from model import VSE
from vocab import Vocabulary
from data import get_transform
from torch.utils import data
from torch.autograd import Variable

def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='runs/coco_vse++_best/model_best.pth.tar')
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/vocab/coco_vocab.pkl')
    parser.add_argument('--crop_size', default=224)
    opt = parser.parse_args()
    print(opt)

    imgdata = im_utils.load(opt.data_root+'/'+opt.image_data)
    labelstrain = im_utils.load(opt.data_root+'/'+opt.labels_train)
    labelstest = im_utils.load(opt.data_root+'/'+opt.labels_test)
    imgmetadata = im_utils.load(opt.data_root+'/'+opt.meta_data)

    # load vocabulary used by the model
    with open(opt.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # load mitstates dataset
    dataset = data_utils.MITstatesDataset(opt.data_root, labelstest,
                                          imgdata, imgmetadata, vocab,
                                          transform=get_transform('coco', 'test', opt))
    dataloader = data.DataLoader(dataset=dataset, batch_size=2, shuffle=False,
                                 collate_fn=data_utils.custom_collate)

    # load model params checkpoint and options
    if torch.cuda.is_available():
        print('compute in GPU')
        checkpoint = torch.load(opt.model_path)
    else:
        print('compute in CPU')
        checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    opt = checkpoint['opt']
    # construct model
    model = VSE(opt)
    # load model state
    model.load_state_dict(checkpoint['model'])

    allobjattsvecs = []
    # TODO remove
    # counter = 0
    for i, objatts in enumerate(dataset.get_all_pairs()):
        print '{} phrases encoded to vectors'.format(i*64)
        if torch.cuda.is_available():
            objatts = objatts.cuda()
        objattsvecs = model.txt_enc(Variable(objatts), [4 for i in range(len(objatts))])
        allobjattsvecs.append(objattsvecs)
        # TODO remove
        # counter += 1
        # if counter == 2:
        #     break
    allobjattsvecs = torch.cat(allobjattsvecs)
    allobjattsvecs = allobjattsvecs.data.cpu().numpy()

    totaltop5 = 0
    totaltop1 = 0
    for i, (images, objatts, lengths, imgids, imgpaths) in enumerate(dataloader):
        print '{}/{} data items encoded'.format(i*2, len(dataloader))

        if torch.cuda.is_available():
            objatts = objatts.cuda()
            images = images.cuda()

        # encode all attribute-object pair phrase on test set
        objattsvecs = model.txt_enc(Variable(objatts), lengths)
        objattsvecs = objattsvecs.data.cpu().numpy()
        # encode all images from test set
        imgvecs = model.img_enc(Variable(images))
        imgvecs = imgvecs.data.cpu().numpy()

        targetdist = np.einsum('ij,ij->i', imgvecs, objattsvecs)
        allobjattdist = np.dot(imgvecs, allobjattsvecs.T)
        sorteddist = np.sort(allobjattdist)
        top5pred = sum(targetdist > sorteddist[:,-5])
        top1pred = sum(targetdist > sorteddist[:,-1])
        totaltop5 += top5pred
        totaltop1 += top1pred

        # TODO remove
        # import pdb; pdb.set_trace()
        # break

    print 'top-5 accuracy: {}, top-1 accuracy: {}'.format(
        float(totaltop5)/len(dataset), float(totaltop1)/len(dataset)
    )

if __name__ == '__main__':
    main()