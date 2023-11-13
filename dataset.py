from lib import *
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T

def text_normalize(text):
    text = text.lower()
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

def build_vocab(corpus, tokenizer):
    for text in corpus:
        yield tokenizer(text)

def getTransform(vocab):
    text_tranform = T.Sequential(
        T.VocabTransform(vocab = vocab),
        T.AddToken(1, begin=True),
        T.AddToken(2, begin=False)
    )
    return text_tranform

def pad_or_truncate(seq, max_length):
    if len(seq) < max_length:
        padding = max_length - len(seq)
        seq += [0]*padding
    elif len(seq) > max_length:
        seq = seq[:max_length]
    return seq

class MyDataset(Dataset):
  def __init__(self, img_lst, corpus, transform, tokenizer, text_transform, max_seq_len, pad_or_truncate):
    super().__init__()

    self.img_lst = img_lst
    self.corpus = corpus
    self.tokenizer = tokenizer
    self.text_transform = text_transform
    self.transform = transform
    self.max_seq_len = max_seq_len
    self.pad_or_truncate = pad_or_truncate
  def __len__(self):
    return len(self.img_lst)

  def __getitem__(self, idx):
    img = Image.open(self.img_lst[idx])
    if self.transform is not None:
      img = self.transform(img)
    label = self.corpus[idx]
    label = self.text_transform(self.tokenizer(label))
    label = self.pad_or_truncate(label, self.max_seq_len)
    return img, torch.tensor(label)
  
def collate_func(batch):
  imgs = []
  in_labels = []
  labels = []
  for sample in batch:
    imgs.append(sample[0])
    in_labels.append(sample[1][:-1])
    labels.append(sample[1][1:])
  imgs = torch.stack(imgs, dim=0)
  in_labels = torch.stack(in_labels, dim = 0)
  labels = torch.stack(labels, dim=0)
  return (imgs, in_labels), labels


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(contrast=0.5, brightness = 0.5),
    transforms.Resize((240,240)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


if __name__ == '__main__':
    PATH = './data/Flicker8k_Dataset'

    img_lst = []
    label_lst = []

    with open('./data/Flickr8k.token.txt') as f:
       lines = f.readlines()
       for line in lines:
          line = line.strip()
          line = line.split('#')
          if line[0].endswith('jpg'):
             img_lst.append(os.path.join(PATH, line[0]))
             label_lst.append(text_normalize(line[1][2:]))
    
    print(len(img_lst))
    print(len(label_lst))


    tokenizer = get_tokenizer('basic_english')

    source_vocab = build_vocab_from_iterator(
       build_vocab(label_lst, tokenizer), 
       specials = ['<pad>', '<sos>', '<eos>', '<unk>'],
       special_first = True
    )

    source_vocab.set_default_index(source_vocab['<unk>'])

    VOCAB_SIZE = len(source_vocab)
    print(VOCAB_SIZE)

    text_transform = getTransform(source_vocab)

    num_train = int(len(img_lst)*0.7)
    num_val = int(len(img_lst)*0.2)
    train_img_lst = img_lst[:num_train]
    train_label_lst = label_lst[:num_train]
    val_img_lst = img_lst[num_train:num_train+num_val]
    val_label_lst = label_lst[num_train:num_train+num_val]
    test_img_lst = img_lst[num_train+num_val:]
    test_label_lst = label_lst[num_train+num_val:]


    train_dataset = MyDataset(train_img_lst, train_label_lst, train_transform, tokenizer, text_transform, 25, pad_or_truncate)
    val_dataset = MyDataset(val_img_lst, val_label_lst, val_transform, tokenizer, text_transform, 25, pad_or_truncate)
    test_dataset = MyDataset(test_img_lst, test_label_lst, val_transform, tokenizer, text_transform, 25, pad_or_truncate)

    train_dataloader = DataLoader(train_dataset, 256, True, collate_fn=collate_func)
    val_dataloader = DataLoader(val_dataset, 256, False, collate_fn=collate_func)
    test_dataloader = DataLoader(test_dataset, 256, False, collate_fn=collate_func)

    it = iter(test_dataloader)
    batch = next(it)
    (img, in_label), label = batch
    print(img[0].shape)
    print(in_label[0].shape, in_label[0])
    print(label[0].shape, label[0])