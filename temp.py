# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""


import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop


from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.layers.merge import add


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#Import keras.<something>.<something>
#tensorflow.keras.<something>.<something>

#开始处理字幕
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


filename = r"C:\Users\cyuan\Desktop\ML_summary\图片字幕自动生成\Flickr8k\Flickr8k.token.txt"

# load descriptions
doc = load_doc(filename)  #载入文件
#print(doc[:300])

#就是将图片和对应的描述储存在字典里面
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split() #根据空格分开
		if len(line) < 2: #就是说这个图片没有描述
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

# parse descriptions
descriptions = load_descriptions(doc)
#print('Loaded: %d ' % len(descriptions))

#数据清洗，全都小写，去掉标点，去掉数字
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
    #这个用法很强
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# clean descriptions
clean_descriptions(descriptions)


#将所有的字生成一个字典，
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
    
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary 这个字典包含字幕中所有的单词
vocabulary = to_vocabulary(descriptions)
#print('Original Vocabulary Size: %d' % len(vocabulary))


#保存descriptions字典变量，每一个图片对应一个字幕（以前是一个图片多个字幕）
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():  #遍历字典
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_descriptions(descriptions, 'descriptions.txt')


#####图片字幕告一段落
#得到所有图片的名称
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):   #分行处理，非常好
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load training dataset  所有的8091个图片地址
filename = r"C:\\Users\\cyuan\Desktop\\ML_summary\\图片字幕自动生成\Flickr8k\\Flickr_8k.trainImages.txt"
train = load_set(filename)
#print('Dataset: %d' % len(train)) #一共有6000个图片

# Below path contains all the images
images = r"C:\\Users\\cyuan\\Desktop\\ML_summary\\图片字幕自动生成\\Flickr8k\\Flicker8k_Dataset\\"
# Create a list of all image names in the directory
img = glob.glob(images+r'*.jpg')  #获取指定目录中的所有文件glob.glob



#现在是找出6000个train图片的地址
# Below file conatains the names of images to be used in train data
train_images_file = r'C:\\Users\\cyuan\\Desktop\ML_summary\\图片字幕自动生成\\Flickr8k\\Flickr_8k.trainImages.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
#strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
# Create a list of all the training images with their full path names
train_img = []

#获取所有训练图片的地址
for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set去掉前面的地址
        train_img.append(i) # Add it to the list of train images


#现在是找出测试图片的地址
# Below file conatains the names of images to be used in test data
test_images_file = 'C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\Flickr_8k.testImages.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images
        
#print(len(test_img))


# load clean descriptions into memory 给所有的字幕加上开头和结尾
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split() #将一行分开
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# descriptions 最后处理的样子
train_descriptions = load_clean_descriptions('descriptions.txt', train)
#print('Descriptions: train=%d' % len(train_descriptions))

#这个函数预处理每一个图片
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)  #这个函数是来自于inception v3，将输入的矩阵预处理
    return x


#加载inception V3模型
# Load the inception v3 model
model = InceptionV3(weights='imagenet')

#创建一个新的模型，就是不要inception V3模型的最后两层
# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)


#得到图片的2048的特征向量
# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec



#只需要计算一次
# Call the funtion to encode all the train images
# This will take a while on CPU - Execute this only once
start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
    
print("Time taken in seconds =", time()-start)


with open("C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\encoded_train_images.pkl", "wb") as encoded_pickle:
    dump(encoding_train, encoded_pickle)
"""


"""
#计算测试图片的特征，只用计算一次
# Call the funtion to encode all the test images - Execute this only once
start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)


# Save the bottleneck test features to disk
with open("C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\encoded_test_images.pkl", "wb") as encoded_pickle:
    dump(encoding_test, encoded_pickle)   #保存成pickle格式


#加载已经计算好的，train_img的特征
train_features = load(open("C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

#这个就是把所有的字幕放在一起，放在一个list里
# Create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
#len(all_train_captions)

#计算单词库，word_counts是所有的单词，vocab是词频高于10的单词
# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
#print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))



#建立两个编码表
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

#增加一个单词，就是空格
vocab_size = len(ixtoword) + 1 # one for appended 0's
vocab_size


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# determine the maximum sequence length 求字幕的最大长度
max_length = max_length(train_descriptions)
#print('Description Length: %d' % max_length)




##现在x2就是一个34维的向量，每一个值就是对应0到1652中的一个数字
                #接下来要把，每一维都扩展到200维，34*200，所以输入因改为 2048+34*200=8848.

#所谓的embadding就是把下标转变成一个固定长度的向量，one-hot编码过于稀疏

"""
# Load Glove vectors
glove_dir = "C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\"
embeddings_index = {} # empty dictionary glove是一个已经训练好的文本
f = open(os.path.join(glove_dir, "glove.6B.200d.txt"), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#print('Found %s word vectors.' % len(embeddings_index))
"""
embeddings_index = load(open("C:\\Users\\cyuan\\.spyder-py3\\Flickr8k\\embeddings_index_1.pkl", "rb"))
#现在将我们的词编码成1652*200的矩阵
embedding_dim = 200
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector



#构建一个lstm神经网络
inputs1 = Input(shape=(2048, ))  #Input():用来实例化一个keras张量
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length, ))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

#这里还可以看模型的总结
#model.summary()

model.layers[2].set_weights([embedding_matrix]) #embedding matrix就是embedding层的参数
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer='adam') #使用交叉熵作为目标函数，梯度下降算法

epochs = 10
number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath





# data generator, intended to be used in a call to model.fit_generator()
                  # 描述字典     图片集合   编码      最大长度     每个批次的图片数
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']  #2048列
            for desc in desc_list:
                # encode the sequence 将字幕编码
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    #这是keras的函数，因为keras只接受固定长度的输入，长度为34
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0] 
                    
                    # encode output sequence这也是一个keras函数,就是生成一个一维编码 1652
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]  #34列
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch: #完成n张图片后就输出一次
                print(len(X1[2]),' ',len(X2[1]),' ', len(y))
                
                yield [[array(X1), array(X2)], array(y)]  #前两个连在一起，是一个生成器，可以一次一次的生成数据
                X1, X2, y = list(), list(), list()
                n=0


#开始训练模型
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("C:\\Users\\cyuan\\.spyder-py3\\" + str(i) + ".h5")




















