from lib import *
from tqdm import tqdm
from dataset import *
from model import ImageCaptioningModel, create_combined_mask
import argparse



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def masked_accuracy(label, pred):
    match = (label == pred).to(device)
    mask = (label != 0).to(device)
    match = match & mask
    return torch.sum(match)/torch.sum(mask)


def train(net, train_dataloader, val_dataloader, criterion, optimizer, epochs):
  net.to(device)
  for epoch in range(1,epochs+1):
    net.train()
    total_loss = 0
    val_loss = 0
    acc = 0
    val_acc = 0
    for (imgs, in_labels), labels in tqdm(train_dataloader):
      imgs = imgs.to(device)
      in_labels = in_labels.to(device)
      combined_mask = create_combined_mask(labels, NUM_HEADS).to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(imgs, in_labels, combined_mask)
      loss = criterion(outputs.transpose(1,2), labels)
      total_loss += loss.item()
      acc += masked_accuracy(labels, outputs.argmax(-1))
      loss.backward()
      optimizer.step()

    print('Epoch {} || epoch loss: {} || accuracy: {}'.format(epoch, total_loss/len(train_dataloader), acc/len(train_dataloader)))
    if (epoch-1) % 5 == 0:
      net.eval()
      for (imgs, in_labels), labels in val_dataloader:
        with torch.no_grad():
          imgs = imgs.to(device)
          in_labels = in_labels.to(device)
          causal_mask = nn.Transformer.generate_square_subsequent_mask(in_labels.shape[1]).to(device)
          labels = labels.to(device)
          outputs = model(imgs, in_labels,causal_mask)
          val_acc += masked_accuracy(labels, outputs.argmax(-1))
          loss = criterion(outputs.transpose(1,2), labels)
          val_loss += loss.item()
      print('Evaluate: val loss: {} || val_acc: {}'.format(val_loss/len(val_dataloader), val_acc/len(val_dataloader)))
      torch.save(net.state_dict(), './data/weights/model_{}_{:.4f}.pt'.format(epoch, val_acc/len(val_dataloader)))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type = str, default= './data/Flicker8k_Dataset')
  parser.add_argument('--captions', type = str, default='./data/Flickr8k.token.txt')
  arg = parser.parse_args()
  PATH = arg.path


  img_lst = []
  label_lst = []

  with open(arg.captions) as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        line = line.split('#')
        if line[0].endswith('jpg'):
            img_lst.append(os.path.join(PATH, line[0]))
            label_lst.append(text_normalize(line[1][2:]))


  tokenizer = get_tokenizer('basic_english')
  source_vocab = build_vocab_from_iterator(
      build_vocab(label_lst, tokenizer), 
      specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
      special_first = True
  )

  source_vocab.set_default_index(source_vocab['<unk>'])


  VOCAB_SIZE = len(source_vocab)
  NUM_HEADS = 8
  EMBEDD_DIM = 512
  FF_DIM = 512
  EPOCHS = 100
  SEQ_LEN = 25
  BATCH_SIZE = 64


  text_transform = getTransform(source_vocab)

  num_train = int(len(img_lst)*0.7)
  num_val = int(len(img_lst)*0.2)
  train_img_lst = img_lst[:num_train]
  train_label_lst = label_lst[:num_train]
  val_img_lst = img_lst[num_train:num_train+num_val]
  val_label_lst = label_lst[num_train:num_train+num_val]
  test_img_lst = img_lst[num_train+num_val:]
  test_label_lst = label_lst[num_train+num_val:]


  train_dataset = MyDataset(train_img_lst, train_label_lst, train_transform, tokenizer, text_transform, SEQ_LEN, pad_or_truncate)
  val_dataset = MyDataset(val_img_lst, val_label_lst, val_transform, tokenizer, text_transform, SEQ_LEN, pad_or_truncate)
  test_dataset = MyDataset(test_img_lst, test_label_lst, val_transform, tokenizer, text_transform, SEQ_LEN, pad_or_truncate)

  train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True, collate_fn=collate_func)
  val_dataloader = DataLoader(val_dataset, BATCH_SIZE, False, collate_fn=collate_func)
  test_dataloader = DataLoader(test_dataset, BATCH_SIZE, False, collate_fn=collate_func)

  weights_path = './data/weights'
  if not os.path.exists(weights_path):
     os.mkdir(weights_path)



  

  model = ImageCaptioningModel(VOCAB_SIZE, NUM_HEADS, EMBEDD_DIM, FF_DIM)
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = torch.optim.Adam(model.parameters(), 1e-4)

  train(model, train_dataloader, val_dataloader, criterion, optimizer, EPOCHS)
 