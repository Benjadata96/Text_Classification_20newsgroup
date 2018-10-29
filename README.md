# Text_Classification_20newsgroup
This project aims to do text classification on some newsgroups using Pytorch to implement a Deep Learning model. The dataset can be found here : http://qwone.com/~jason/20Newsgroups/. 

I use the Google Word2Vec pre-trained embedding to represent the words (disponible here: https://code.google.com/archive/p/word2vec/). Using a MLP to do the classification. Since the dataset does not give us the order of the word in each sample, we use the mean vector for each sample (taking the sum of each embedding divided by the number of word).

The 'Datas_20News' file is to create the inputs.
The 'Model_Training_20News' file is to create the model.
The 'Script_20News' file is the script to run the whole process !
