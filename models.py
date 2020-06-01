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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, dim_size=14):
        super(Encoder, self).__init__()
        resnet = tv.models.resnet101(pretrained=True)

        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)

        self.pool = nn.AdaptiveAvgPool2d((dim_size, dim_size))
        
        
    def forward(self, images):
        # [batch_size, encoded_dim_size, encoded_dim_size, 2048]
        return self.pool(self.resnet(images)).permute(0, 2, 3, 1) 

    
class Attention(nn.Module):
    def __init__(self, dim_encoder, dim_decoder, dim_attention):
        super(Attention, self).__init__()
        
        self.attention_encoder = nn.Linear(dim_encoder, dim_attention)
        self.attention_decoder = nn.Linear(dim_decoder, dim_attention)
        self.both = nn.Linear(dim_attention, 1)
    
    def forward(self, out_encoder, hidden_decoder):
        attention_encoder = self.attention_encoder(out_encoder)
        attention_decoder = self.attention_decoder(hidden_decoder)
        
        weights = self.both(torch.relu(attention_encoder + attention_decoder.unsqueeze(1))).squeeze(2)
        weights = torch.softmax(weights, dim=1)
        
        out = torch.sum((out_encoder * weights.unsqueeze(2)), dim=1)
        
        return out, weights 

class Decoder(nn.Module):
    def __init__(self, dim_attention, dim_embed, dim_decoder, vocab_size, dim_encoder=2048):
        super(Decoder, self).__init__()
        
        self.dim_encoder = dim_encoder
        self.dim_attention = dim_attention
        self.dim_embed = dim_embed
        self.vocab_size = vocab_size
        
        self.attention = Attention(dim_encoder, dim_decoder, dim_attention)
        self.embed = nn.Embedding(vocab_size, dim_embed)
        self.drop = nn.Dropout(p=0.5)
        
        self.decode_lstm = nn.LSTMCell(dim_embed + dim_encoder, dim_decoder, bias=False)
        self.h_init = nn.Linear(dim_encoder, dim_decoder)
        self.c_init = nn.Linear(dim_encoder, dim_decoder)
        self.f = nn.Linear(dim_decoder, dim_encoder)
        
        self.fc1 = nn.Linear(dim_decoder, vocab_size)
        
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        
    def init_hidden(self, out_encoder):
        out = out_encoder.mean(dim=1)
        
        return self.h_init(out), self.c_init(out)
    
    def forward(self, out_encoder, captions, lengths):
        batch_size = out_encoder.size(0)
        dim_encoder = out_encoder.size(-1)
        vocab_size = self.vocab_size
        
        out_encoder = out_encoder.view(batch_size, -1, dim_encoder)
        pixels = out_encoder.size(1)
        
        lengths, ind = lengths.squeeze(1).sort(dim=0, descending=True)
        out_encoder = out_encoder[ind]
        captions = captions[ind]
        
        embed = self.embed(captions)
        
        # init hidden state
        h, c = self.init_hidden(out_encoder)
        
        lengths = (lengths-1).tolist()
        
        predict = torch.zeros(batch_size, max(lengths), vocab_size).to(device)
        weights = torch.zeros(batch_size, max(lengths), pixels).to(device)
        
        for time_step in range(max(lengths)):
            batch_t = sum([i > time_step for i in lengths])
            
            weighted_encoder, alpha = self.attention(out_encoder[:batch_t], h[:batch_t])
            
            sig = torch.sigmoid(self.f(h[:batch_t]))
            weighted_encoder = sig * weighted_encoder
            
            h, c = self.decode_lstm(torch.cat([embed[:batch_t, time_step, :], weighted_encoder], dim=1),
                                    (h[:batch_t], c[:batch_t]))
            
            output = self.fc1(self.drop(h))
            predict[:batch_t, time_step, :] = output
            weights[:batch_t, time_step, :] = alpha
            
        return predict, captions, lengths, weights, ind
    
    def caption(self, encoded_img_features, word_map, beam_size=5):
        '''
            In this method, given the encoded image features, we will do beam 
            search to obtain the final captions for the image. 
        '''
        print(encoded_img_features.shape)
        
        enc_img_size, encoder_dimension = encoded_img_features.size(1), encoded_img_features.size(3)
        encoded_img_features = encoded_img_features.view(1, -1, encoder_dimension)
        total_pixels = encoded_img_features.size(1)
        
        encoded_img_features = encoded_img_features.expand(beam_size, total_pixels, encoder_dimension)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #initialize all sequences to only contain the start token at very beginning 
        captions =  torch.LongTensor([[word_map['<START>']]] * beam_size).to(device)

        prev_words = captions
        top_scores = torch.zeros(beam_size, 1).to(device)
        
        
        # alpha weights 
        soft_weights = torch.ones(beam_size, 1, enc_img_size, enc_img_size).to(device)
        
        completed_captions = []
        completed_captions_scores = []
        completed_captions_weights = []
        
        current_iter = 1 
        h,c = self.init_hidden(encoded_img_features)
        
        while True: 
#             import pdb; pdb.set_trace()
            embedding = self.embed(prev_words).squeeze(1)
            weighted_encoder, alpha = self.attention(encoded_img_features, h)
            
            sig = torch.sigmoid(self.f(h))
            weighted_encoder = sig * weighted_encoder
            
            h,c = self.decode_lstm(torch.cat((embedding, weighted_encoder), dim=1), (h,c))
            output = self.fc1(h)
            output = F.log_softmax(output, dim=1)
            
            output = top_scores.expand_as(output) + output 
            assert output.size(1) == len(word_map)
            
            if current_iter == 1: 
                top_scores, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_scores, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs, next_word_idxs = top_words / output.size(1), top_words % output.size(1)
            
            
            captions = torch.cat((captions[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            
            
            alpha = alpha.view(-1, enc_img_size, enc_img_size)
            soft_weights = torch.cat((soft_weights[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete_idx = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_map['<END>']]
            complete_idx = list(set(range(len(next_word_idxs))) - set(incomplete_idx))
            
            if len(complete_idx):
                print("extending caption")
                completed_captions.extend(captions[complete_idx].tolist())
                completed_captions_scores.extend(top_scores[complete_idx])
                completed_captions_weights.extend(soft_weights[complete_idx].tolist())
            
            beam_size -= len(complete_idx)
            
            #done with beam search and can break
            if beam_size == 0: 
                print("finished beam search")
                break 
            
            captions = captions[incomplete_idx]
            soft_weights = soft_weights[incomplete_idx]
            h, c = h[prev_word_idxs[incomplete_idx]], c[prev_word_idxs[incomplete_idx]]
            encoded_img_features = encoded_img_features[prev_word_idxs[incomplete_idx]]
            top_scores = top_scores[incomplete_idx].unsqueeze(1)
            prev_words = next_word_idxs[incomplete_idx].unsqueeze(1)

            if current_iter > 50: 
                print("max iteration complete")
                break
            
            current_iter+=1 
        
        print(completed_captions_scores)
        
        best_seq_idx = completed_captions_scores.index(max(completed_captions_scores))
        chosen_caption, caption_weight = completed_captions[best_seq_idx], completed_captions_weights[best_seq_idx]
        
        
        return chosen_caption, caption_weight,   
    
    def visualize_attention(image, original_image, decoder, encoder_image_features, smooth_weights = True):
        caption, weights = decoder.caption(encoder_image_features, word_map)
        idx2word = {v: k for k, v in word_map.items()}  
            
        words = [idx2word[i] for i in caption]
        
        for i in range(len(words)):
            if (i>50):
                break
            
            plt.subplot(np.ceil(len(words)/5.), 5, (i+1))
            plt.text(0, 1, '%s' % (words[i]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(original_image)
                    
            current_weight = weights[i]
            
            if smooth_weights:
                weight = skimage.transform.pyramid_expand(np.asarray(current_weight), upscale=24, sigma=8)
            else:
                weight = skimage.transform.resize(np.asarray(current_weight), [14*24, 14*24])        
            
            if(i==0):
                plt.imshow(weight, alpha=0)
            else:
                plt.imshow(weight, alpha=0.8)
            
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        
        plt.show()   
