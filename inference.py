from lib import *
from model import ImageCaptioningModel
from dataset import *
import matplotlib.pyplot as plt
from PIL import Image
import random
import argparse

def generate(model, img, seq_len, transform):
    model.eval()
    sentence = torch.tensor([[1]])

    transformed_img = transform(img).unsqueeze(0)
    with torch.no_grad():
        for i in range(seq_len):
            causal_mask = nn.Transformer.generate_square_subsequent_mask(sentence.shape[1])
            result = model(transformed_img, sentence, causal_mask)

            sentence = torch.concat([sentence, result.argmax(-1)[:,i].unsqueeze(0)], dim=1)
        
        text = ''
        for i in sentence[0]:
            if i == 2:
                text += source_vocab.get_itos()[i] 
                break
            text += source_vocab.get_itos()[i] + ' '

        return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weights', default='./data/weights/model_11_0.4076.pt')
    parser.add_argument('-g','--generate_from_test', type=bool, default=False, help='evaluate base on test dataset')
    parser.add_argument('-i', '--image', default=None, help='input image path to evaluate')
    parser.add_argument('--path', default='./data/Flicker8k_Dataset')
    arg = parser.parse_args()
    PATH = arg.path
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

    model = ImageCaptioningModel(VOCAB_SIZE, NUM_HEADS, EMBEDD_DIM, FF_DIM)
    model.load_state_dict(torch.load(arg.pretrained_weights, map_location=torch.device('cpu')))

    #Evaluate base on test dataset (get 6 random images and generates captions)
    if arg.generate_from_test == True:
        fig, axes = plt.subplots(6,1,figsize=(10,25))
        for i in range(6):
            idx = random.randint(0, len(test_img_lst))
            img = Image.open(test_img_lst[idx])
            text = generate(model, img, SEQ_LEN, val_transform)
            axes[i].imshow(img)
            axes[i].set_title(text)
        plt.show()


    #Inference for input image
    if arg.image is not None:
        img = Image.open(arg.image)

        sentence = torch.tensor([[1]])
        transformed_img = val_transform(img).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            for i in range(SEQ_LEN):
                causal_mask = nn.Transformer.generate_square_subsequent_mask(sentence.shape[1])
                result = model(transformed_img, sentence, causal_mask)

                sentence = torch.concat([sentence, result.argmax(-1)[:,i].unsqueeze(0)], dim=1)
        
            text = ''
            for i in sentence[0]:
                if i == 2:
                    text += source_vocab.get_itos()[i] 
                    break
                text += source_vocab.get_itos()[i] + ' '

        plt.imshow(img)
        plt.title(text)
        plt.show()