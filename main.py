# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 12:07:16 2018

@author: Shinelon
"""
import numpy as np
import copy
import time

def simi(x, y):
    return np.dot(x,y)/(np.sqrt(np.sum(np.square(x)))*np.sqrt(np.sum(np.square(y))))

class corpus():
    def __init__(self):
        self.userToId = {}
        self.itemToId = {}
        self.pos_per_user = {}
        self.itemFeat = {}
        self.nUsers = 0
        self.nItems = 0
        self.posEvent = []
        self.temp = {}
        
    def load_ratedata(self, filename):
        f = open(filename, 'r')
        for i, l in enumerate(f):
            if i == 0:
                continue
            user, item = l.strip().split()[:2]
            if not self.userToId.has_key(user):
                self.userToId[user] = self.nUsers
                self.nUsers += 1
            if not self.itemToId.has_key(item):
                self.itemToId[item] = self.nItems
                self.nItems += 1
            if not self.pos_per_user.has_key(self.userToId[user]):
                self.pos_per_user[self.userToId[user]] = [self.itemToId[item]]
            else:
                self.pos_per_user[self.userToId[user]].append(self.itemToId[item])
        f.close()
        
    def load_itemfeat(self, actorfilename, countryfilename, genrefilename):
        # actor features
        f = open(actorfilename, 'r')
        for i, l in enumerate(f):
            if i == 0:
                continue
            l = l.strip().split()
            item = l[0]
            actor = int(l[-1])-1
            if actor >= 40:
                continue
            if not self.itemToId.has_key(item):
                continue
            if not self.itemFeat.has_key(self.itemToId[item]):
                self.itemFeat[self.itemToId[item]] = [0]*40
                self.itemFeat[self.itemToId[item]][actor] = 1
            else:
                self.itemFeat[self.itemToId[item]][actor] = 1
        f.close()
        # country features
        countryToId = {}
        nCountries = 0
        f = open(countryfilename, 'r')
        for i, l in enumerate(f):
            if i == 0:
                continue
            try:
                item, country = l.strip().split('\t')
            except:
                continue
            if not self.itemToId.has_key(item):
                continue
            if not countryToId.has_key(country):
                countryToId[country] = nCountries
                nCountries += 1
            if not self.temp.has_key(self.itemToId[item]):
                self.temp[self.itemToId[item]] = [countryToId[country]]
            else:
                self.temp[self.itemToId[item]].append(countryToId[country])
        f.close()
        for i in self.temp:
            l = [0]*nCountries
            for w in self.temp[i]:
                l[w] = 1
            if self.itemFeat.has_key(i):
                self.itemFeat[i].extend(l)
        self.temp = {}
        countryToId = {}
        # genres features
        genreToId = {}
        nGenres = 0
        f = open(genrefilename, 'r')
        for i, l in enumerate(f):
            if i == 0:
                continue
            try:
                item, genre = l.strip().split('\t')
            except:
                continue
            if not self.itemToId.has_key(item):
                continue
            if not genreToId.has_key(genre):
                genreToId[genre] = nGenres
                nGenres += 1
            if not self.temp.has_key(self.itemToId[item]):
                self.temp[self.itemToId[item]] = [genreToId[genre]]
            else:
                self.temp[self.itemToId[item]].append(genreToId[genre])
        f.close()
        for i in self.temp:
            l = [0]*nGenres
            for w in self.temp[i]:
                l[w] = 1
            if self.itemFeat.has_key(i):
                self.itemFeat[i].extend(l)
        self.temp = {}
        genreToId = {}
        #clean data
        for w in self.itemFeat:
            if len(self.itemFeat[w]) != 131:
                self.itemFeat[w].extend([0]*(131-len(self.itemFeat[w])))
        return
        
class BPRMF():
    def __init__(self, corp, K, iterations, learn_rate, reg):
        self.pos_per_user = corp.pos_per_user
        self.nUsers = corp.nUsers
        self.nItems = corp.nItems
        self.K = K
        self.gamma_user = np.random.randn(self.nUsers, self.K)
        self.gamma_item = np.random.randn(self.nItems, self.K)
        self.beta_item = np.random.randn(self.nItems, 1)
        
    def init(self):
        self.test_per_user = {}
        self.nItems -= 2000
        for u in self.pos_per_user:
            temp = []
            for w in self.pos_per_user[u]:
                if w < self.nItems:
                    temp.append(w)
            self.pos_per_user[u] = copy.deepcopy(temp[:-1])
            self.test_per_user[u] = temp[-1]
        self.gamma_user = np.random.randn(self.nUsers, self.K)
        self.gamma_item = np.random.randn(self.nItems, self.K)
        self.beta_item = np.random.randn(self.nItems, 1)
        
    def train(self, iterations, learn_rate, reg):
        print "Training......"
        t = 0
        for i in range(iterations):
            if i%100000 == 0:
                print "100000 iterations took %f seconds"%(time.time()-t)
                t = time.time()   
            self.oneiteration(learn_rate, reg)
        
    def oneiteration(self, learn_rate, reg):
        userId = self.sampleUser()
        pos_list = self.pos_per_user[userId]
        posItemId = pos_list[np.random.randint(0, len(pos_list))]
        while True:
            negItemId = self.sampleItem()
            if negItemId not in pos_list:
                break
        self.updateFactors(userId, posItemId, negItemId, learn_rate, reg)
        
       
    def sampleUser(self):
        return np.random.randint(0, self.nUsers)
    
    def sampleItem(self):
        return np.random.randint(0, self.nItems)
    
    def updateFactors(self, userId, posItemId, negItemId, learn_rate, reg):
        uij = np.dot(self.gamma_user[userId], self.gamma_item[posItemId]) - np.dot(self.gamma_user[userId], self.gamma_item[negItemId])
        uij += self.beta_item[posItemId] - self.beta_item[negItemId]
        deri = np.exp(-uij)/(1 + np.exp(-uij))
        u = self.gamma_user[userId]
        i = self.gamma_item[posItemId]
        j = self.gamma_item[negItemId]
        self.gamma_user[userId] += learn_rate*(deri*(i-j)-reg*u)
        self.gamma_item[posItemId] += learn_rate*(deri*u-reg*i)
        self.gamma_item[negItemId] += learn_rate*(-deri*u-reg*j)
        self.beta_item[posItemId] += learn_rate*(deri-reg*self.beta_item[posItemId])
        self.beta_item[negItemId] += learn_rate*(-deri-reg*self.beta_item[negItemId])
        
    def auc(self):
        count = 0.
        pos = 0.
        score = []
        for u in self.test_per_user:
            posItemId = self.test_per_user[u]
            xui = self.prediction(u, posItemId)
            for j in range(self.nItems):
                if self.pos_per_user.has_key(j) or j == posItemId:
                    continue
                xuj = self.prediction(u, j)
                if xui > xuj:
                    pos += 1.
                count += 1.
            score.append(pos/count)
        auc = sum(score)/len(score)
        return auc
    
    def prediction(self, u, i):
        return np.dot(self.gamma_user[u], self.gamma_item[i]) + self.beta_item[i]
        
        
class MAP():
    def __init__(self, pos_per_user , test_per_user, gamma_user, M, itemFeat):
        self.test_per_user = test_per_user
        self.pos_per_user = pos_per_user
        self.M = M
        self.L = len(itemFeat[0])
        self.K = len(gamma_user[0])
        self.N = len(gamma_user)
        self.gamma_user = gamma_user
        self.itemFeat = itemFeat
        self.W = np.zeros((self.K, self.L))
        
    def train(self, iterations, learn_rate, reg):
        print "Maping......"
        t = 0
        for i in range(iterations):
            if i%100000 == 0:
                print "100000 iterations took %f seconds"%(time.time()-t)
                t = time.time()   
            self.oneiteration(learn_rate, reg)
    
    def oneiteration(self, learn_rate, reg):
        userId = self.sampleUser()
        pos_list = self.pos_per_user[userId]
        while True:
            posItemId = pos_list[np.random.randint(0, len(pos_list))]
            if self.itemFeat.has_key(posItemId):
                break
        while True:
            negItemId = self.sampleItem()
            if negItemId not in pos_list and self.itemFeat.has_key(negItemId):
                break
        self.updateFactors(userId, posItemId, negItemId, learn_rate, reg)
        
    def sampleUser(self):
        return np.random.randint(0, self.N)
    
    def sampleItem(self):
        return np.random.randint(0, self.M)
    
    def updateFactors(self, userId, posItemId, negItemId, learn_rate, reg):
        try:
            uij = np.dot(np.dot(self.gamma_user[userId], self.W), self.itemFeat[posItemId]) - np.dot(np.dot(self.gamma_user[userId], self.W), self.itemFeat[negItemId])
            deri = np.exp(-uij)/(1 + np.exp(-uij))
            self.W += learn_rate*(deri*np.array([self.gamma_user[userId]]).T*(np.array([self.itemFeat[posItemId]])-np.array([self.itemFeat[negItemId]]))-reg*self.W)
        except:
            print len(self.gamma_user[userId]), len(self.itemFeat[posItemId]), len(self.itemFeat[negItemId])
    
    def auc(self):
        count = 0.
        pos = 0.
        score = []
        for u in self.test_per_user:
            posItemId = self.test_per_user[u]
            if not self.itemFeat.has_key(posItemId):
                continue
            xui = self.prediction(u, posItemId)
            for j in range(self.M):
                if self.pos_per_user.has_key(j) or j == posItemId or not self.itemFeat.has_key(j):
                    continue
                xuj = self.prediction(u, j)
                if xui > xuj:
                    pos += 1.
                count += 1.
            score.append(pos/count)
        auc = sum(score)/len(score)
        return auc
    
    def prediction(self, u, i):
        return np.dot(self.gamma_user[u], np.dot(self.W, self.itemFeat[i]))


                
        
            
corp = corpus()
corp.load_ratedata("C:/Users/Shinelon/Desktop/hetrec2011-movielens-2k-v2/user_ratedmovies.dat")
corp.load_itemfeat("C:/Users/Shinelon/Desktop/hetrec2011-movielens-2k-v2/movie_actors.dat", 
                   "C:/Users/Shinelon/Desktop/hetrec2011-movielens-2k-v2/movie_countries.dat",
                   "C:/Users/Shinelon/Desktop/hetrec2011-movielens-2k-v2/movie_genres.dat")
model = BPRMF(corp, 20, 200, 0.05, 0.01)
#x =  copy.deepcopy(model.beta_item)
model.init()
model.train(3000000, 0.01, 0.01)
x = model.auc()
#print x[:5],'\n', model.beta_item[:5]
#model.nItems
Map = MAP(model.pos_per_user, model.test_per_user, model.gamma_user, model.nItems, corp.itemFeat)
Map.train(4000000, 0.01, 0.01)
y = Map.auc()

