#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:44:40 2018

@author: BenjaminSalem
"""
import numpy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

class Net(nn.Module): 
    def __init__(self,nins,nout):
        super(Net, self).__init__()
        self.nins=nins
        self.nout=nout
        nhid1 = (nins+nout)/2
        self.hidden = nn.Linear(int(nins), int(nhid1))
        self.hidden2= nn.Linear(int(nhid1), int(nhid1))
        self.out    = nn.Linear(int(nhid1), int(nout))
        
    def forward(self, x):
        dropout = nn.Dropout(p=0.5)
        x1 = dropout(F.relu(self.hidden(x)))
        x2 = dropout(F.relu(self.hidden2(x1)))
        x_out = self.out(x2)
        return (x_out)
    

class training_model():
    
    def __init__(self, gradient_step, nb_epochs, model):
        self.accuracy = None
        self.gdt_step = gradient_step
        self.nb_epochs = nb_epochs
        self.model = model
    
    def compute_accuracy(self,model,data,target):
        
        x = Variable(torch.FloatTensor(data))
        y = model(x).data.numpy()
        
        
        haty = numpy.argmax(y,axis=1)
        nok = sum([1 for i in range(len(target)) if target[i] == haty[i]])
        
        self.accuracy = float(nok) / float(len(target))
        
        return (self.accuracy)
    
    def model_training(self,X_train,X_val,Y_train,Y_val):
                
        training_acc = []
        validation_acc = []
        training_loss = []
        validation_loss = []
        
        optim = torch.optim.Adam(self.model.parameters(), lr = self.gdt_step)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.nb_epochs):
            optim.zero_grad()
            
            haty_train = self.model(Variable(torch.FloatTensor(X_train)))
            acc_train = self.compute_accuracy(self.model,X_train,Y_train)
            training_acc.append(acc_train)
            loss_train = criterion(haty_train,Variable(torch.LongTensor(Y_train)))
            training_loss.append(float(loss_train.data.numpy()))
            loss_train.backward()
            optim.step()
            
            haty_val = self.model(Variable(torch.FloatTensor(X_val)))
            acc_val = self.compute_accuracy(self.model, X_val, Y_val)
            validation_acc.append(acc_val)
            loss_val = criterion(haty_val, Variable(torch.LongTensor(Y_val)))
            validation_loss.append(float(loss_val.data.numpy()))
            
            print('accuracy : '+str(acc_train) + '  // loss : '+str(loss_train.data.numpy()[0]))
            if epoch >= 1:
                if validation_loss[epoch] < validation_loss[epoch-1]:
                    torch.save(self.model.state_dict(), ".dernier_modele_sauv")
                    print ('.. new model saved ..')
            
        return(training_acc, validation_acc, training_loss, validation_loss)
                    
    def testing_model(self, X_test, Y_test):
                
        self.model.load_state_dict(torch.load(".dernier_modele_sauv"))
        
        criterion = nn.CrossEntropyLoss()
        
        haty_test = self.model(Variable(torch.FloatTensor(X_test)))
        acc_test = self.compute_accuracy(self.model, X_test, Y_test)
        loss_test = criterion(haty_test, Variable(torch.LongTensor(Y_test)))
        print('Test_accuracy : ' +str(acc_test) +' / Test_Loss : ' +str(loss_test))

    def metrics_plotting(self,accuracy_list,loss_list, training_or_valid) :
        
        plt.figure()
        plt.plot(range(self.nb_epochs), accuracy_list, label = 'Accuracy_'+str(training_or_valid))
        plt.legend()
        plt.title(str(training_or_valid)+' Accuracy with a step of '+str(self.gdt_step) )
        plt.show()
        
        plt.figure()
        plt.plot(range(self.nb_epochs), loss_list, label = 'Loss_'+str(training_or_valid))
        plt.legend()
        plt.title(str(training_or_valid)+' Loss with a step of '+str(self.gdt_step) )
        plt.show()
        
    def main(self, X_train, X_val, X_test, Y_train, Y_val, Y_test):
        
        training_acc, validation_acc, training_loss, validation_loss = self.model_training(X_train, X_val, Y_train, Y_val)
        self.testing_model(X_test,Y_test)
        self.metrics_plotting(training_acc, training_loss, 'Training')
        self.metrics_plotting(validation_acc, validation_loss, 'Validation')