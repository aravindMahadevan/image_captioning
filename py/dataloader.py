import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
import random as random
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

image_folder = 'data/Flickr8k_Dataset'
karpathy_json_path = 'data/dataset_flickr8k.json'
MAX_LENGTH = 50

def get_train_val_test_splits(image_folder, image_splits, max_caption_len, min_word_freq):
    '''
        method will return three lists containing the training, validation, and testing 
        file paths and captions. Method will also return the word map that will be used for captioning
        images. 
        Parameters: 
            image_folder: location of the images in the data set, either MSCOCO, Flickr8K, Flickr30k 
            
            image_splits: dictionary that contains the captions for each image and indicates whether 
            image part of training, validation, or testing set
            
            max_caption_len: threshold for maximum caption length 
            
            min_word_freq: threshold that determines whether a word will be in word map or not. 
            
            
        Output:
            train_img_caps: list of tuples containing the training image file path and caption
            val_img_caps: list of tuples containing the validation image file path and caption
            test_img_caps: list of tuples containing the testing image file path and caption
        '''
    #storing tuple of path to img and the caption
    word_freq = Counter()
    train_img_caps = []
    val_img_caps = []
    test_img_caps = [] 
    num_train_img, num_val_img, num_test_img = 0, 0, 0 
    for img in image_splits['images']:
        img_captions = []
        for word in img['sentences']:
            #check if the caption length is not to long
            if len(word['tokens']) <= max_caption_len:
                img_captions.append(word['tokens'])
            # Update word frequency
            word_freq.update(word['tokens'])

        #if caption is of length zero move to next image 
        if not len(img_captions): 
            continue 

        img_file_path = os.path.join(image_folder, img['filename'])
        #save corresponding files and captions 
        if img['split'] == 'train':
            train_img_caps.append((img_file_path, img_captions))
            num_train_img+=1
        elif img['split'] == 'val':
            val_img_caps.append((img_file_path, img_captions))
            num_val_img+=1 
        elif img['split'] == 'test':
            test_img_caps.append((img_file_path, img_captions))
            num_test_img+=1
    
    #create a limited vocabulary and don't include any word that hasn't appeared 
    #min_word_freq times
    words = [w for w in word_freq if word_freq[w] >= min_word_freq]
    min_words = [w for w in word_freq if word_freq[w] < min_word_freq]
    word_map = {word: i+1 for i, word in enumerate(words)}
    #specify start token, end token, unknown token, and padding token 
    word_map['<START>'] = len(word_map) + 1 
    word_map['<END>'] = len(word_map) + 1
    word_map['<UNK>'] = len(word_map) + 1
    word_map['<PAD>'] = 0
    
    print("Number of training images: {0}".format(num_train_img))
    print("Number of validation images: {0}".format(num_val_img))
    print("Number of testing images: {0}".format(num_test_img))
    return train_img_caps, val_img_caps, test_img_caps, word_map

def create_dataset(data, split, word_map, base_file_name, captions_per_image):
    output_folder = 'data/'
    encoded_captions = []
    encoded_captions_length = []
    start_token = word_map['<START>']
    end_token = word_map['<END>']
    unknown_token = word_map['<UNK>']
    padding_token = word_map['<PAD>']
    training_data_file = os.path.join(output_folder, base_file_name + '_' + split + '_images.hdf5')
    encoded_captions_file = os.path.join(output_folder, base_file_name + '_' + split + '_encoded_captions.json')
    encoded_captions_length_file = os.path.join(output_folder, base_file_name + '_' + split + '_encoded_caption_lengths.json')
    
    print("Creating %s data set" % split)
    with h5py.File(os.path.join(output_folder, base_file_name + '_' + split + '_images' + '.hdf5'), 'a') as h:
        try:
            images = h.create_dataset('images', (len(data), 3, 256, 256), dtype='uint8')
        except:
            print("Already created dataset. Exiting from this method")
            return        

        for image_idx ,(image_path, image_captions) in enumerate(data):    
            #want to ensure that there are at least certain number of captions per image 
            #if current image has less than that threshold, then augement the captions
            num_captions = len(image_captions)
            if num_captions < captions_per_image: 
                chosen_captions = [random.choice(image_captions) for _ in range(captions_per_image - num_captions)]
                chosen_captions += image_captions
            else:
                chosen_captions = random.sample(image_captions, k = captions_per_image)
            
            #for the chosen captions, encode them
            
            for i, caption in enumerate(chosen_captions):
                encoded_caption = [word_map.get(w,unknown_token) for w in caption]
                assert len(caption) == len(encoded_caption)
                padding_for_caption = [padding_token for _ in range(MAX_LENGTH- len(caption))]
                encoded_caption = [start_token] + encoded_caption + [end_token] + padding_for_caption
                
                encoded_captions.append(encoded_caption)
                

                assert len(encoded_caption) == MAX_LENGTH + 2 
                encoded_captions_length.append(len(caption) + 2)
            
            #resize all images to be 256 x 256 
            image = Image.open(image_path)
            image_resize = image.resize((256, 256))
            image_array = np.asarray(image_resize).transpose(2,0,1) #ensures that 3x256x256
            images[image_idx] = image_array
            
            
            assert len(image_array.shape) == 3
            
        h.attrs['cpi'] = captions_per_image 
        
        print("Saving the encoded captions")
        #save the encoded captions and the encoded caption lengths to a json file 
        with open(encoded_captions_file, 'w') as j:
            json.dump(encoded_captions, j)

        with open(encoded_captions_length_file, 'w') as j:
            json.dump(encoded_captions_length, j)
        
        print("Done creating the dataset for split ")

class MyDataset(Dataset):
    def __init__(self, folder, name, split, transform=None):
        '''
            Create a data set class that will be used when passing into the data loader. 
        '''
        self.split = split
        
        self.file = h5py.File(os.path.join(folder, name + '_' + self.split + '_images.hdf5'))
        self.images = self.file['images']
        
        self.cpi = self.file.attrs['cpi']
        
        # load captions
        with open(os.path.join(folder, name + '_' + self.split + '_encoded_captions.json'), 'r') as f:
            self.captions = json.load(f)
            
        # load captions' lenghts
        with open(os.path.join(folder, name + '_' + self.split + '_encoded_caption_lengths.json'), 'r') as f:
            self.lengths = json.load(f)
        
                        
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx // self.cpi] / 255.0)
        
        #TODO: not using standard formulation of mean=0, std=1
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        caption, caption_length = torch.LongTensor(self.captions[idx]), torch.LongTensor([self.lengths[idx]])
        
        if self.split == 'train':
            return image, caption, caption_length
        else:
            start = self.cpi * (idx // self.cpi)
            end = start + self.cpi
            
            captions = torch.LongTensor(self.captions[start:end])
            
            
            return image, caption, caption_length, captions
        
    def __len__(self):
        return len(self.captions)
        

def init_dataloader():
  #get the training splits across all data set 
  with open(karpathy_json_path, 'r') as j:
      flickr8k_datasplit = json.load(j)

  train_data, val_data, test_data, word_map = get_train_val_test_splits(image_folder, flickr8k_datasplit, 50, 5)
  
  create_dataset(train_data, 'train', word_map, 'flickr8k', 5)
  create_dataset(test_data, 'test', word_map, 'flickr8k', 5)
  create_dataset(val_data, 'val', word_map, 'flickr8k', 5)

  train_set = MyDataset('data', 'flickr8k', 'train')
  val_set = MyDataset('data', 'flickr8k', 'val')
  test_set = MyDataset('data', 'flickr8k', 'test')

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=16,shuffle=True, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, pin_memory=True)

  return word_map, train_loader, val_loader, test_loader

if __name__ == '__main__':
  init_dataloader()


  

