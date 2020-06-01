from dataloader import *
from models import *

class ImageCaptioning():
  def __init__(self, output_dir, num_epochs=100):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.word_map, self.train_loader, self.val_loader, self.test_loader = init_dataloader()

    self.encoder = Encoder().to(self.device)
    self.decoder = Decoder(dim_attention=512, dim_embed=512, dim_decoder=512, vocab_size=len(self.word_map)).to(self.device)

    self.optimizer_parameter_group = [{'params' : self.encoder.parameters()},
                                      {'params' : self.decoder.parameters()}]
    self.criterion = nn.CrossEntropyLoss().to(self.device)

    self.optimizer = torch.optim.Adam(self.optimizer_parameter_group, lr=4e-4)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5)


    self.batch_size = self.train_loader.batch_size
    self.num_epochs = num_epochs
    self.history = []
    self.train_loss = []
    self.val_loss = []
    self.bleu1, self.bleu2, self.bleu3, self.bleu4 = [], [], [], []

    os.makedirs(output_dir, exist_ok=True)
    self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
    self.config_path = os.path.join(output_dir, "config.txt")

    if os.path.isfile(self.config_path):
#       with open(self.config_path, 'r') as f:
#         if f.read()[:-1] != repr(self):
#             print("Cannot create this experiment: found a conflicting checkpoint with current setting")
# #           raise ValueError("Cannot create this experiment: found a conflicting checkpoint with current setting")
      print("Loading existing checkpoint.")
      self.load()
    else:
      self.save()

  @property
  def epoch(self):
    return len(self.history)
  
  def setting(self):
    return {'Encoder' : self.encoder,
            'Decoder' : self.decoder,
            'Optimizer' : self.optimizer,
            'BatchSize' : self.batch_size}
  
  def __repr__(self):
    string = ''
    for key, val in self.setting().items():
      string += '{}({})\n'.format(key, val)
    return string
  
  def state_dict(self):
    return {'Encoder' : self.encoder.state_dict(),
            'Decoder' : self.decoder.state_dict(),
            'Optimizer' : self.optimizer.state_dict(),
            'History' : self.history,
            'TrainLoss' : self.train_loss,
            'ValLoss' : self.val_loss ,
            'Bleu1' : self.bleu1, 
            'Bleu2' : self.bleu2, 
            'Bleu3' : self.bleu3, 
            'Bleu4' : self.bleu4}

  def load_state_dict(self, checkpoint):
    self.encoder.load_state_dict(checkpoint['Encoder'])
    self.decoder.load_state_dict(checkpoint['Decoder'])
    self.optimizer.load_state_dict(checkpoint['Optimizer'])

    self.history = checkpoint['History']
    self.train_loss = checkpoint['TrainLoss']
    self.val_loss = checkpoint['ValLoss']
    self.bleu1 = checkpoint['Bleu1']
    self.bleu2 = checkpoint['Bleu2']
    self.bleu3 = checkpoint['Bleu3']
    self.bleu4 = checkpoint['Bleu4']

    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(self.device)

  def save(self):
    torch.save(self.state_dict(), self.checkpoint_path)
    with open(self.config_path, 'w') as f:
      print(self, file=f)
  
  def load(self):
    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
    self.load_state_dict(checkpoint)
    del checkpoint

  def run(self):
    for epoch in range(self.num_epochs):
#       self.scheduler.step()
      self.train(epoch)
      self.validate()
  
  def train(self, epoch):
    self.decoder.train()
    self.encoder.train()
    
    loss_sum = 0
    loss_num = 0
    
    for i, (images, captions, lengths) in enumerate(self.train_loader):
        images = images.to(self.device)
        captions = captions.to(self.device)
        lengths = lengths.to(self.device)

        img_features = self.encoder(images)
        predict, captions, lengths, weights, ind = self.decoder(img_features, captions, lengths)
        targets = captions[:,1:]
        
        predict = pack_padded_sequence(predict, lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, lengths, batch_first=True).data
        
        loss = self.criterion(predict, targets)
        
        loss += ((1. - weights.sum(dim=1)) ** 2).mean()
        
        self.optimizer.zero_grad()

        loss.backward()
        
        self.optimizer.step()
        
        loss_sum += loss.item()*sum(lengths)
        loss_num += sum(lengths)

        if i % 100 == 0: 
            print("Epoch % d [%d/%d], Loss: %f" % (epoch, i, len(self.train_loader), loss_sum/loss_num))
            self.save()

    self.train_loss.append(loss_sum/loss_num)
    self.save()

  def validate(self):
      self.decoder.eval()
      self.encoder.eval()
      
      refs = []
      hypos = []
      
      loss_sum = 0
      loss_num = 0
      
      with torch.no_grad():
          for i, (images, captions, lengths, all_captions) in enumerate(self.val_loader):
              images = images.to(self.device)
              captions = captions.to(self.device)
              lengths = lengths.to(self.device)
              
              images = self.encoder(images)
              predict, captions, lengths, weights, ind = self.decoder(images, captions, lengths)
              
              targets = captions[:,1:]
              
              scores = predict.clone()
              
              predict = pack_padded_sequence(predict, lengths, batch_first=True).data
              targets = pack_padded_sequence(targets, lengths, batch_first=True).data
              
              loss = self.criterion(predict, targets)
              
              loss += ((1. - weights.sum(dim=1)) ** 2).mean()
                      
              loss_sum += loss.item()*sum(lengths)
              loss_num += sum(lengths)
                      
              if (i % 100) == 0:
                  print("Validate [%d/%d], Loss: %f" % (i, len(self.val_loader), loss_sum/loss_num))
              
              all_captions = all_captions[ind]
              for j in range(all_captions.shape[0]):
                  image_captions = all_captions[j].tolist()
                  image_captions = list(map(lambda c : [w for w in c if w not in {self.word_map['<START>'], 
                                                                                  self.word_map['<PAD>']}],
                                          image_captions))
                  refs.append(image_captions)
              
              pred = torch.max(scores, dim=2)[1].tolist()
              temp = []
              for j, p in enumerate(pred):
                  temp.append(pred[j][:lengths[j]])
              pred = temp
              hypos.extend(pred)
              
              assert len(refs) == len(hypos)
                
          bleu_1 = corpus_bleu(refs, hypos, weights=(1, 0, 0, 0))
          bleu_2 = corpus_bleu(refs, hypos, weights=(0.5, 0.5, 0, 0))
          bleu_3 = corpus_bleu(refs, hypos, weights=(0.33, 0.33, 0.33, 0))
          bleu_4 = corpus_bleu(refs, hypos)

          print("BLEU-1: %f", bleu_1)
          print("BLEU-2: %f", bleu_2)
          print("BLEU-3: %f", bleu_3)
          print("Validate. BLEU-4: %f, Loss: %f" % (bleu_4, loss_sum/loss_num))

          self.val_loss.append(loss_sum/loss_num)
          self.bleu1.append(bleu_1)
          self.bleu2.append(bleu_2)
          self.bleu3.append(bleu_3)
          self.bleu4.append(bleu_4)


exp_name = 'exp_batch32_bi_false'
exp = ImageCaptioning(output_dir=exp_name)
exp.run()



  



    





