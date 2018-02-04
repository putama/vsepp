import os
import pickle
import argparse

from compositionutils import im_utils
from compositionutils import data_utils

from vocab import Vocabulary
from data import get_transform
from torch.utils import data

def main():
    print('evaluate vse on visual composition...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/mitstates_data')
    parser.add_argument('--image_data', default='mit_image_data.pklz')
    parser.add_argument('--labels_train', default='split_labels_train.pklz')
    parser.add_argument('--labels_test', default='split_labels_test.pklz')
    parser.add_argument('--meta_data', default='split_meta_info.pklz')
    parser.add_argument('--vocab_path', default='data/coco_vocab.pkl')
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

    for i, (images, objatt_tensors, imgids, imgpaths) in enumerate(dataloader):
        print i

    # encode all attribute-object pair phrase on test set
    # encode all images from test set

    print 'done'

if __name__ == '__main__':
    main()