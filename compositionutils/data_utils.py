import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class MITstatesDataset(data.Dataset):
    '''Custom dataset implementation of MITstates data'''
    def __init__(self, root, splitdata, imgdata, imgmetadata, vocab, transform=None):
        self.root = root
        self.splitdata = splitdata
        self.imgdata = imgdata
        self.imgmetadata = imgmetadata
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        # load image
        imgid = self.splitdata['imIds'][index]
        imgpath = self.imgdata['images'][imgid]['file_name']
        image = Image.open(os.path.join(self.root, 'images', imgpath.replace('_',' ',1))).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # load att-obj phrase
        objatt_id = self.imgdata['annotations'][imgid]['pair_labs'][0]
        objatt_name = self.imgmetadata['pairNames'][objatt_id]
        objatt_tensor = self.text_to_ids(objatt_name)

        if objatt_tensor is None:
            return None, None, None, None

        return image, objatt_tensor, imgid, imgpath

    def text_to_ids(self, phrase):
        ids = [] # word/token ids
        ids.append(self.vocab('<start>'))
        tokens = phrase.lower().split('_')
        for token in tokens:
            tokenid = self.vocab(token)
            if tokenid == self.vocab('<unk>'):
                return None
            ids.append(tokenid)
        ids.append(self.vocab('<end>'))
        ids_tensor = torch.Tensor(ids)
        return ids_tensor

    def get_all_pairs(self):
        pairs = []
        for pairName in self.imgmetadata['pairNames']:
            pair = self.text_to_ids(pairName)
            if pair is not None:
                pairs.append(pair)
        # TODO remove
        # group_size = 4
        group_size = 64
        grouppedpairs = []
        for i in range(0,len(pairs),group_size):
            grouppedpairs.append(torch.stack(pairs[i:i+group_size], 0).long())
        return grouppedpairs

    def __len__(self):
        return len(self.splitdata['imIds'])

def custom_collate(items):
    try:
        tmp = items
    	items = filter(lambda x: x[0] is not None, items)
    	images, objatt_tensors, imgids, imgpaths = zip(*items)

    	lengths = [len(phrase) for phrase in objatt_tensors]

    	# stack images and objatt phrase into a batch
    	images = torch.stack(images, 0)
    	objatt_tensors = torch.stack(objatt_tensors, 0).long()

    	return images, objatt_tensors, lengths, imgids, imgpaths
    except:
        return None, None, None, None, None	
