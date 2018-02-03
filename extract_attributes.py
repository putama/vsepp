import argparse
import pickle
import os
import nltk
import spacy

from vocab import Vocabulary
from pycocotools.coco import COCO

import data

def main():
    print('extract attributes from captions...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='./data/vocab',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    opt = parser.parse_args()
    print(opt)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    paths = data.get_paths(opt.data_path+'/'+opt.data_name)[0]
    coco = COCO(paths['train']['cap'])

    nlp = spacy.load('en')
    attributes2nouns = {}
    attributes2count = {}
    for i, key in enumerate(coco.anns.keys()):
        caption = coco.anns[key]['caption']
        doc = nlp(str(caption).lower().decode('utf-8'))
        for token in doc:
            if token.dep_ == 'amod':
                if not attributes2count.has_key(token.text):
                    attributes2count[token.text] = 0
                if not attributes2nouns.has_key(token.text):
                    attributes2nouns[token.text] = set()

                attributes2count[token.text] = attributes2count[token.text] + 1
                attributes2nouns[token.text].add(token.head.text)

        if i % 100 == 0:
            print ('{}/{} captions processed. {} attributes'
                    .format(i, len(coco.anns.keys()), len(attributes2count)))

    k = 1000
    # extract top k attributes
    attributes = sorted(attributes2count, key=attributes2count.get, reverse=True)[0:k]
    # filter attributes2nouns
    attributes2nouns = {att: attributes2nouns[att] for att in attributes}

    writepath = opt.data_path+'/'+opt.data_name+'_attributes.pkl'
    pickle.dump({
        "attributes": attributes,
        "attributes2nouns": attributes2nouns},
        open(writepath, "wb")
    )
    print('extraction finished and written to {}'.format(writepath))

if __name__ == '__main__':
    main()