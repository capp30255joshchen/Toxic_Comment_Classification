import numpy as np
np.random.seed(29)

import sys
import gensim
from collections import Counter
import re

import pandas as pd
from unidecode import unidecode
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU, Lambda
from keras.preprocessing import text, sequence

from keras.callbacks import Callback
from keras import optimizers
from keras import backend

import gc

class Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        
        self.interval = interval
        self.X_valid, self.y_valid = validation_data
        self.max_score = 0
        self.no_improvement_counts = 0
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_valid, verbose=1)
            score = roc_auc_score(self.y_valid, y_pred)
            print("\n ROC-AUC: epoch: %d, score: %.4f \n" % (epoch+1, score))
            if score > self.max_score:
                print("New High Score! (previous: %.4f) \n" % self.max_score)
                self.model.save_weights("best_weights.h5")
                self.max_score = score
                self.no_improvement_counts = 0
            else:
                self.no_improvement_counts += 1
                ## early stop if we fail to see any improvement after 3 epochs of training
                if self.no_improvement_counts > 3:
                    self.model.stop_training = True
                    print("Epoch %d: early stopping, high score = %.4f" % (epoch, self.max_score))
                    print("No improvements. Early stop. High score = %.4f" % self.max_score)


# This code is based on: Spellchecker using Word2vec by CPMP
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
class SpellChecker(object):
    def __init__(self, fasttext_file, twitter_file):
        self.fasttext_file = fasttext_file
        self.twitter_file = twitter_file
        self.spell_model = gensim.models.KeyedVectors.load_word2vec_format(fasttext_file)
        self.words = self.spell_model.index2word
        self.WORDS = {}
        
        for rank, word in enumerate(self.words):
            self.WORDS[word] = rank
    
    
    def get_embeddings(self):
        def get_coef(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        
        embed_idx_ft = dict(get_coef(*ft.rstrip().rsplit(' ')) for ft in open(EMBEDDING_FILE_FASTTEXT, encoding='utf-8'))
        embed_idx_tw = dict(get_coef(*tw.strip().split()) for tw in open(EMBEDDING_FILE_TWITTER, encoding='utf-8'))
        return embed_idx_ft, embed_idx_tw
    
    def convert_words(self, text):
        return re.findall(r'\w+', text.lower())
    
    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)
    
    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
    
    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def singlify(self, word):
        return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])



class NLPModel(object):
    def __init__(self, train_file_path, test_file_path, clipvalue=1, num_filters=40, dropout=0.5, embed_size=501):
        self.train = pd.read_csv(train_file_path)
        self.test = pd.read_csv(test_file_path)
        self.clipvalue = clipvalue
        self.num_filters = num_filters
        self.dropout = dropout
        self.embed_size = embed_size
        
        ## set by cleanNonASCII()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        
        ## set by augmentDataset()
        self.train_features = None
        self.test_features = None
        
        ## set by tokenizeData()
        self.pad_X_train = None
        self.pad_X_test = None
        self.word_index = None
        self.max_features = None
        self.maxlen = None
        
        ## set by createEmbedMatrix()
        self.embedding_matrix = None
        
    
    def cleanNonASCII(self):
        special_char_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]', re.IGNORECASE)
        def clean_text(x):
            x_ascii = unidecode(x)
            x_clean = special_char_removal.sub('', x_ascii)
            return x_clean
        self.train['clean_text'] = self.train['comment_text'].apply(lambda x: clean_text(str(x)))
        self.test['clean_text'] = self.test['comment_text'].apply(lambda x: clean_text(str(x)))
        
        ## convert to numpy representation by calling .values 
        # or use 'something' instead of 'unknown'?
        self.X_train = self.train['clean_text'].fillna("unknown").values 
        self.X_test = self.test['clean_text'].fillna("unknown").values 
        self.y_train = self.train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values 
    
    def augmentDataset(self):
        
        def add_features(df):
            df['comment_text'] = df['comment_text'].apply(lambda x: str(x))
            df['total_length'] = df['comment_text'].apply(len)
            ## number of capitals in one comment
            df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
            df['caps_len_ratio'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']), axis=1)
            df['total_words'] = df['comment_text'].str.count('\S+')
            df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
            df['unique_total_ratio'] = df['num_unique_words'] / df['total_words']
            return df
        
        self.train = add_features(self.train)
        self.test = add_features(self.test)
        
        self.train_features = self.train[['caps_len_ratio', 'unique_total_ratio']].fillna(0)
        self.test_features = self.test[['caps_len_ratio', 'unique_total_ratio']].fillna(0)
        
        ## standardize
        ss = StandardScaler()
        ss.fit(np.vstack((self.train_features, self.test_features)))
        self.train_features = ss.transform(self.train_features)
        self.test_features = ss.transform(self.test_features)
    
    def tokenizeData(self, max_features = 50000, maxlen = 100):
        '''
        len(tokenizer.word_index) gives a dictionary of total number of unique words ranked from the most 
        important (most frequent) to the least.
        
        max_features denote how many important words to keep. 
        e.g. max_features=10000 keeps top 10000 most frequent words
        
        texts_to_sequences: array of top 10000 most frequent words' ranking that appeared for each comment
        
        pad_sequences: Pads sequences to the same length (=maxlen) -- pads shorter seq with 0
        '''
        self.max_features = max_features
        self.maxlen = maxlen
        
        tokenizer = text.Tokenizer(num_words = max_features)
        tokenizer.fit_on_texts(list(self.X_train) + list(self.X_test))
        self.word_index = tokenizer.word_index
        print("total number of unique words: ", len(tokenizer.word_index))
        
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = tokenizer.texts_to_sequences(self.X_test)
        self.pad_X_train = sequence.pad_sequences(X_train_seq, maxlen=maxlen)
        self.pad_X_test = sequence.pad_sequences(X_test_seq, maxlen=maxlen)
        
        
        
    
    def createEmbedMatrix(self, fasttext_file, twitter_file):
        
        ## instantiate SpellChecker
        spellchecker = SpellChecker(fasttext_file, twitter_file)
        embeddings_index_ft, embeddings_index_tw = spellchecker.get_embeddings()
        print("Got embeddings_index_ft and _tw")
        
        rows_words = min(self.max_features, len(self.word_index))
        self.embedding_matrix = np.zeros((rows_words, self.embed_size))
        print("Done read in fasttext, twitter")
        
        unknown = np.zeros((self.embed_size, ))
        unknown[:300, ] = embeddings_index_ft.get("unknown")
        unknown[300:500, ] = embeddings_index_tw.get("unknown")
        unknown[500, ] = 0
        
        def all_caps(word):
            ## True if the input word is all capitalized
            return len(word) > 1 and word.isupper()
        
        def embed_word(word, word_idx, embedding_matrix):
            embed_vec_ft = embeddings_index_ft.get(word)
            if embed_vec_ft is not None:
                end_value = np.array([1]) if all_caps(word) else np.array([0])
                embedding_matrix[i, :300] = embed_vec_ft
                embedding_matrix[i, 500] = end_value
                
                embed_vec_tw = embeddings_index_tw.get(word)
                if embed_vec_tw is not None:
                    embedding_matrix[i, 300:500] = embed_vec_tw
        
        for word, i in self.word_index.items():
            if i >= self.max_features:
                continue
            if embeddings_index_ft.get(word) is not None:
                embed_word(word, i, self.embedding_matrix)
            else:
                if len(word) > 20:
                    self.embedding_matrix[i] = unknown
                else:
                    word_alter = spellchecker.correction(word)
                    if embeddings_index_ft.get(word_alter) is not None:
                        embed_word(word_alter, i, self.embedding_matrix)
                    else:
                        word_alter = spellchecker.correction(spellchecker.singlify(word))
                        if embeddings_index_ft.get(word_alter) is not None:
                            embed_word(word_alter, i, self.embedding_matrix)
                        else:
                            self.embedding_matrix[i] = unknown
    
    def build_model(self):
    	'''
    	6-layer NN:
    		1. concat fasttext and glove twitter word embeddings.
    		2. set up dropout to prevent overfitting
    		3. Bidirectional LSTM
    		4. Bidirectional GRU
    		5. concat average pooling, the last state, max pooling, and "all-caps words ratio", "unique words ratio"
    		6. ouput dense layer with sigmoid (relu not so good) 
    	optimizer: Adam --> less sensitive to learning rate
    	'''

        features_input = Input(shape=(self.train_features.shape[1], ))
        input_ = Input(shape=(self.maxlen, ))
        x = Embedding(self.max_features, self.embed_size, weights=[self.embedding_matrix], trainable=False)(input_)

        x = SpatialDropout1D(self.dropout)(x)

        x = Bidirectional(LSTM(self.num_filters, return_sequences=True))(x)

        x, x_h, x_c = Bidirectional(GRU(self.num_filters, return_sequences=True, return_state=True))(x)

        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = concatenate([avg_pool, x_h, max_pool, features_input])
        
        output_ = Dense(6, activation='sigmoid')(x) 
        
        model = Model(inputs=[input_, features_input], outputs=output_)
        adam = optimizers.adam(clipvalue = self.clipvalue)
        
        #model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model



if __name__ == "__main__":
	####
	# Please download train_processed.csv and test_processed.csv from
	# https://www.kaggle.com/xbf6xbf/processing-helps-boosting-about-0-0005-on-lb
	# Credits to xbf : Processing helps boosting about 0.0005 on LB
	# 
	# Please download the following files
	# crawl-300d-2M.vec 
	# glove.twitter.27B.200d.txt
	# sample_submission.csv

	# This function should be called with 5 arguments:
    #    sys.argv[1]: Comma-delimited file containing preprocessed train data (train_processed.csv)
    #    sys.argv[2]: Comma-delimited file containing preprocessed test data (test_processed.csv)
    #    sys.argv[3]: Comma-delimited file containing FastText (crawl-300d-2M.vec)
    #    sys.argv[4]: Comma-delimited file containing Glove Twitter (glove.twitter.27B.200d.txt)
    #    sys.argv[5]: Comma-delimited file containing sample submission (sample_submission.csv)
    if (len(sys.argv) != 6):
        print("usage: NLPmodel.py train_processed.csv test_processed.csv crawl-300d-2M.vec glove.twitter.27B.200d.txt sample_submission.csv")
        sys.exit(2)

    train_path = sys.argv[1]
	test_path = sys.argv[2]
	EMBEDDING_FILE_FASTTEXT = sys.argv[3]
	EMBEDDING_FILE_TWITTER = sys.argv[4]
	sample_submission_path = sys.argv[5]


	NLPmodel = NLPModel(train_path, test_path)
	NLPmodel.cleanNonASCII()
	print("finish cleanNonASCII")

	NLPmodel.augmentDataset()
	print("finish augmentDataset")

	NLPmodel.tokenizeData()
	print("finish tokenizeData")

	NLPmodel.createEmbedMatrix(EMBEDDING_FILE_FASTTEXT, EMBEDDING_FILE_TWITTER)
	print("finish createEmbedMatrix")


	batch_size = 32
	epochs = 1
	num_folds = 2
	predict = np.zeros((NLPmodel.test.shape[0] ,6))

	x_train = NLPmodel.pad_X_train
	y_train = NLPmodel.y_train
	features = NLPmodel.train_features
	x_test = NLPmodel.pad_X_test
	test_features = NLPmodel.test_features

	kfolds = KFold(n_splits=num_folds, shuffle=True, random_state=29)

	for train_idx, test_idx in kfolds.split(x_train):
	    kfold_y_train, kfold_y_test = y_train[train_idx], y_train[test_idx]
	    kfold_X_train = x_train[train_idx]
	    kfold_X_features = features[train_idx]
	    kfold_X_valid = x_train[test_idx]
	    kfold_X_valid_features = features[test_idx]

	    gc.collect()
	    backend.clear_session()

	    model = NLPmodel.build_model()

	    ra_val = Evaluation(validation_data=([kfold_X_valid,kfold_X_valid_features],kfold_y_test), interval=1)
	    
	    print("start fitting")
	    model.fit([kfold_X_train,kfold_X_features], kfold_y_train, 
	    			batch_size=batch_size, epochs = epochs, verbose=1, callbacks=[ra_val])
	    gc.collect()

	    model.load_weights("best_weights.h5")
	    predict += model.predict([x_test, test_features],batch_size=batch_size,verbose=1) / num_folds

	print("Done")

	# output the prediction
	sample_submission = pd.read_csv(sample_submission_path)
	label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	sample_submission[label_cols] = predict
	sample_submission.to_csv("model_submission.csv", index=False)