import string
import time
import numpy as np
import os
from pickle import dump, load
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical , get_file 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional, add

from tqdm import tqdm
tqdm().pandas()

def load_doc(filename):
        file= open(filename, 'r')
        text= file.read() 
        file.close()
        return text

def all_img_captions(filename):
        file= load_doc(filename)
        captions=file.split('\n') # it 's used to split the text into lines'
        descriptions={}
        for caption in captions[:-1]:
                img,  caption= caption.split('\t')
                if img[:-2] not in descriptions:
                        descriptions[img[:-2]]=[caption]
                else:
                        descriptions[img[:-2]].append(caption)

        return descriptions

def cleaning_text(captions):
        table= str.maketrans('', '', string.punctuation)
        for img, caps in captions.items():
                for i, img_captions in enumerate(caps):
                    img_captions.replace("-","")
                    desc=img_captions.split()
                    desc=[word.lower() for word in desc]
                    desc=[w.translate(table) for w in desc]
                    desc=[word for word in desc if len(word)>1]
                    desc=[word for word in desc if word.isalpha()]

                    img_caption=' '.join(desc)
                    captions[img][i]=img_caption


        return captions

def text__vocabulary(descriptions):
        vocab= set()
        for key in descriptions.keys():
                [vocab.update(d.split()) for d in descriptions[key]]
        return vocab

def save_descriptions(descriptions, filename):
        lines=list()
        for key, desc_list in descriptions.items():
                for desc in desc_list:
                        lines.append(key + '\t' + desc)
        data='\n'.join(lines)
        file=open(filename, 'w')
        file.write(data)
        file.close()

dataset_text="Flickr8k_text"
dataset_images="Flicker8k_Dataset"

filename= dataset_text + "/Flickr8k.token.txt"
descriptions=all_img_captions(filename)
print("Total number of images in the dataset: ", len(descriptions))

cleaned_descriptions= cleaning_text(descriptions)
vocabulary= text__vocabulary(cleaned_descriptions)
print("Total number of unique words in the vocabulary: ", len(vocabulary))

save_descriptions(cleaned_descriptions, "descriptions.txt")                

def download_with_retry(url,filename, max_retries=3):
        for attempt in range(max_retries):
                try:
                        return get_file(filename,url)
                except Exception as e:
                        if(attempt==max_retries-1):
                                raise e
                        print("Download attempt failed")
                        time.sleep(3)

# weights_url="https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
# weights_path = download_with_retry(weights_url,'xception_weights.h5')
# model = Xception(include_top=False, pooling="avg", weights=weights_path)

def extract_features(directory):
        features={}
        valid_images= {".jpg", ".jpeg", ".png"}
        for img in tqdm(os.listdir(directory)):
                ext=os.path.splitext(img)[1].lower()
                if ext not in valid_images: 
                        continue
                filename = directory + "/" + img
                image=Image.open(filename)
                image=image.resize((299, 299))
                image=np.expand_dims(image, axis=0)
                image= image/127.5
                image= image-1.0
                feature= model.predict(image)
                features[img]=feature

        return features
# features= extract_features(dataset_images)
# with open("features.pkl", "wb") as file:
#         dump(features, file)

# features= load(open("features.pkl", "rb"))

def load_photos(filename):
        file= load_doc(filename)
        photos=file.split('\n')[:-1]
        photos_present=[photo for photo in photos if os.path.isfile(os.path.join(dataset_images, photo))]

        return photos_present

def load_clean_descriptions(filename, photos):
        file=load_doc(filename)
        descriptions={}
        for line in file.split('\n'):
          words=line.split()
          if len(words)<1:
                continue

          image,image_caption=words[0], words[1:]
          if image in photos:
                if image not in descriptions:
                        descriptions[image]=[]
                desc='<start> ' + ' '.join(image_caption) + ' <end>'
                descriptions[image].append(desc)

        return descriptions
def load_features(photos):
        all_features= load(open("features.pkl", "rb"))
        features={k: all_features[k] for k in photos}

        return features

filename= dataset_text + "/Flickr_8k.trainImages.txt"
train_imgs= load_photos(filename)
train_descriptions= load_clean_descriptions("descriptions.txt",train_imgs)
train_features= load_features(train_imgs)

def dict_to_list(descriptions):
        all_desc=[]
        for key in descriptions.keys():
                [all_desc.append(d) for d in descriptions[key]]
        return all_desc 

def create_tokenizer(descriptions):
        desc_list=dict_to_list(descriptions)
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(desc_list)
        return tokenizer

tokenizer = create_tokenizer(train_descriptions)

# dump(tokenizer, open("tokenizer.pkl", "wb"))

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size: ", vocab_size)

def max_length(descriptions):
        desc_list=dict_to_list(descriptions)
        return max(len(d.split()) for d in desc_list)

max_length = max_length(train_descriptions)
print("Maximum Description Length: ", max_length)


def data_generator(descriptions, features, tokenizer, max_length, num_photos_per_batch):
        def generator():
                while True:             
                        for  key,descriptions_list in descriptions.items():
                                feature=features[key][0]
                                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, descriptions_list, feature)
                                for i in range(len(input_image)):
                                        yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]

        output_signature = (
                {
                        'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
                        'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
                },
                tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        return dataset.batch(32)

def create_sequences(tokenizer, max_length, desc_list, features):
        X1,X2,y=list(), list(), list()
        for desc in desc_list:
                seq=tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                        in_seq, out_seq=seq[:i], seq[i]
                        in_seq=pad_sequences([in_seq], maxlen=max_length)[0]
                        out_seq=to_categorical([out_seq], num_classes=vocab_size)[0]
                        X1.append(features)
                        X2.append(in_seq)
                        y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)
dataset = data_generator(train_descriptions, train_features, tokenizer, max_length,num_photos_per_batch=32)
for(a,b) in dataset.take(1):
        print(a['input_1'].shape, a['input_2'].shape, b.shape)
        break

def define_model(vocab_size, max_length):
        inputs1=Input(shape=(2048,), name='input_1')
        fel=Dropout(0.5)(inputs1)
        fe2=Dense(256, activation='relu')(fel)

        inputs2=Input(shape=(max_length,), name='input_2')
        se1=Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2=Dropout(0.5)(se1)
        se3=LSTM(256)(se2)

        decoder1=add([fe2, se3])
        decoder2=Dense(256, activation='relu')(decoder1)
        outputs=Dense(vocab_size, activation='softmax')(decoder2)
        model=Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())
        return model

model=define_model(vocab_size, max_length)

epochs=10

steps_per_epoch=50

os.makedirs("models")
for i in range(epochs):
        dataset = data_generator(train_descriptions, train_features, tokenizer, max_length, num_photos_per_batch=32)
        model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
        model.save("models/model_" + str(i) + ".h5")