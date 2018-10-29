import numpy
import codecs
import gensim
from nltk.corpus import stopwords 
import os.path
from sklearn.model_selection import train_test_split

List_stopwords = set(stopwords.words('english'))
WordEmb = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

def dataset_vocabulary_index(dataset_vocab_path):
        
        with codecs.open(dataset_vocab_path,"r","utf-8") as vocab_file:
            vocabulary = {}
            line = 1 
            
            for l in vocab_file:
                s = l.strip().split()
                vocabulary[line] = s[0]
                line += 1
        print ('.. dataset vocabulary dict created ..')
        return (vocabulary)

dataset_voc = dataset_vocabulary_index('vocabulary.txt')

class Input_Data():
    
    def __init__(self, datas_repo):
        self.datas_repo = datas_repo
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        self.validation_split = 0.35
        
    
    def sample_vocabulary(self, file, dataset_vocab):
        sample_voc = {}
        with codecs.open(os.path.join(self.datas_repo,file),"r","utf-8") as training_file:
            
            lines = training_file.readlines()
            for i in lines : 
                sample, index_word, nb_appearance = i.strip().split()
                word = dataset_vocab[int(index_word)]
                if word in WordEmb.vocab and word not in List_stopwords:
                    if sample_voc.get(sample) is None:
                        sample_voc[sample] = {}
                    
                    sample_voc[sample][word]=int(nb_appearance)
                else :
                    i=i
        print('.. sample vocabulary dict created ..')
        return (sample_voc)
    
    def create_output(self, file_y, is_training):
        Y_gold = []
        with codecs.open(os.path.join(self.datas_repo,file_y),"r","utf-8") as label_file:
            
            for j in label_file:
                s = j.strip().split()
                label = int(s[0]) - 1 #going from 0 to 19 instead of 1 to 20
                Y_gold.append(label)
            
            Y_gold = numpy.array(Y_gold, dtype = numpy.int64)
            if len(Y_gold) > 8000:
                Y_gold = numpy.delete(Y_gold,10728) #only stopwords sample
            if len(Y_gold) < 8000:
                Y_gold = numpy.delete(Y_gold,6964) #only stopwords sample
        return(Y_gold)

    
    def create_input(self,sample_vocabulary, is_training):
        X = numpy.zeros((len(sample_vocabulary),300))
        i = 0 
        for key, value in sample_vocabulary.items():
            total_word_per_sample = 0
            score = 0 
            
            for k,v in value.items():
                total_word_per_sample += v
                score += v*(WordEmb[k])
            X[i] = (score / total_word_per_sample) 
            i+=1
        
        X = numpy.array(X, dtype = numpy.float32)
        return(X)


    def training_input(self,file,file_y, is_training, dataset_voc_dict):

        dataset_vocab = dataset_voc_dict
        sample_vocab = self.sample_vocabulary(file,dataset_vocab)

        if is_training == 1:
            self.X_train = self.create_input(sample_vocab, is_training)
            self.Y_train = self.create_output(file_y, is_training)
            return(self.X_train, self.Y_train, sample_vocab)
        
        else:
            self.X_test = self.create_input(sample_vocab, is_training)
            self.Y_test = self.create_output(file_y, is_training)
            self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(self.X_test,self.Y_test, test_size=self.validation_split, random_state=13)
            return (self.X_test, self.X_val, self.Y_test, self.Y_val)

