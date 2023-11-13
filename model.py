from lib import *
from dataset import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cnn_model():
    base_model = models.efficientnet_b1(weights = models.EfficientNet_B1_Weights.DEFAULT)
    dct = {'features': 'features'}
    model = IntermediateLayerGetter(base_model, dct)
    i = 0
    for params in model.parameters():
        i += 1
        if i <= 270:
            params.requires_grad_(False)
    return model

def create_combined_mask(groundtruths, n_heads):
    batch_size, seq_len = groundtruths.shape[0], groundtruths.shape[1]
    causal_mask = torch.triu(torch.ones((batch_size, seq_len, seq_len)), diagonal=1)

    mask = (groundtruths == torch.zeros_like(groundtruths)).unsqueeze(1)

    combined_mask = torch.maximum(causal_mask, mask)
    combined_mask = combined_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).reshape(-1, seq_len, seq_len).type(torch.bool)
    return combined_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedd = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.rand(100, embed_dim))
        self.embed_scale = math.sqrt(embed_dim)

    def forward(self, inputs):
        length = inputs.shape[-1]
        embed_tokens = self.embedd(inputs) * self.embed_scale
        embed_positions = self.position[:inputs.shape[-1]]

        return embed_positions + embed_tokens


class TransformerEncoderLayer(nn.Module):
  def __init__(self, n_heads, embed_dim, dim_ff):
    super().__init__()
    self.n_heads = n_heads
    self.embed_dim = embed_dim
    self.dim_ff = dim_ff

    self.attn_1 = nn.MultiheadAttention(self.embed_dim, self.n_heads, dropout=0.1, batch_first=True)
    self.layernorm1 = nn.LayerNorm(1280)
    self.layernorm2 = nn.LayerNorm(self.embed_dim)
    self.linear1 = nn.Linear(1280, self.embed_dim)
    self.relu1 = nn.ReLU()

  def forward(self, inputs):
    inputs = self.layernorm1(inputs)
    inputs = self.relu1(self.linear1(inputs))
    output, weights = self.attn_1(query=inputs,key=inputs,value=inputs)
    output = self.layernorm2(inputs + output)

    return output



class TransformerDecoderLayer(nn.Module):
  def __init__(self, vocab_size, embed_dim, ff_dim, num_heads):
    super().__init__()
    self.embed_dim = embed_dim
    self.ff_dim = ff_dim
    self.num_heads = num_heads

    self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0., batch_first = True)

    self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0., batch_first=True)

    self.ffn_layer_1 = nn.Linear(embed_dim, ff_dim)
    self.relu1 = nn.ReLU()

    self.ffn_layer_2 = nn.Linear(ff_dim, embed_dim)

    self.layernorm1 = nn.LayerNorm(embed_dim)
    self.layernorm2 = nn.LayerNorm(embed_dim)
    self.layernorm3 = nn.LayerNorm(embed_dim)

    self.embedding = PositionalEmbedding(vocab_size, embed_dim)

    self.dropout1 = nn.Dropout(0.3)
    self.dropout2 = nn.Dropout(0.3)
    self.final_layer = nn.Linear(embed_dim, vocab_size)

  
  def forward(self, inputs, encoder_outputs, combined_mask):

    inputs = self.embedding(inputs)

    attn_1, scores = self.attention_1(query=inputs, value=inputs, key=inputs,
                              attn_mask=combined_mask, average_attn_weights=False)

    out_1 = self.layernorm1(inputs + attn_1)

    attn_2, _ = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs)
    out_2 = self.layernorm2(out_1 + attn_2)

    ffn_out = self.ffn_layer_1(out_2)
    ffn_out = self.dropout1(ffn_out)
    ffn_out = self.ffn_layer_2(ffn_out)

    ffn_out = self.layernorm3(ffn_out + out_2)
    ffn_out = self.dropout2(ffn_out)
    preds = self.final_layer(ffn_out)
    return preds

class ImageCaptioningModel(nn.Module):
  def __init__(self, vocab_size, num_heads, embed_dim, dim_ff):
      super().__init__()
      self.cnn_backbone = get_cnn_model()
      self.encoder = TransformerEncoderLayer(num_heads, embed_dim, dim_ff)
      self.decoder = TransformerDecoderLayer(vocab_size, embed_dim, dim_ff, num_heads)
      
  def forward(self, imgs, inputs, mask):
      img_embedd = self.cnn_backbone(imgs)['features']
      img_embedd = nn.Flatten(2)(img_embedd).transpose(1,2)

      encoder_output = self.encoder(img_embedd)

      output = self.decoder(inputs, encoder_output, mask)
      return output

train_transform = transforms.Compose([
    transforms.Resize((300,300)),
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
  

  model = ImageCaptioningModel(VOCAB_SIZE, NUM_HEADS, EMBEDD_DIM, FF_DIM)
  model = model.to(device)
  img = img.to(device)
  in_label = in_label.to(device)
  combined_mask = create_combined_mask(label, 8)
  label = label.to(device)

  model(img, in_label, combined_mask.to(device))