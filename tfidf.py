import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import re
import pickle
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.preprocessing import normalize
from collections import Counter



class TFIDF:

    """
    An object that learns statistics about the ngrams in a collection of strings
    to produce TFIDF and complex TFIDF embeddings.

    Parameters
    ----------
        gramlen: the length of ngrams to use.
        verbose: if True, will print time bars to estimate how long some processes willt take.
        spaces: Add spaces to either end of strings when processing. 
        normalize: If true, tfidf and complex tfidf representations will be l2 normalized.

    The class is designed to be used with dataframes of short strings, along with an
    optional column of counts or integer weights for each string (as if you obtained
    these by a group-by in SQL.)

    Example
    ----------
        df = pd.DataFrame({"text":["bob","dole"], "counts":[100,20]})
        tfidf = TFDIF(2)
        tfidf.fit(df,desc_col = "text", counts_col = "counts)

        #The transform argument can take either a pandas series or list.

        tfidf.transform(["bob","dole"])

        #There is also a complex tfidf transform, see the notebook for details. 
        #it also takes an argument $N$ intended to be the max length of the strings 
        #being considered. It's default value is 10.

        tfidf.complex_transform(["bob","dole"], N=10)

        #One can also view the values of the components of the (unnormalized) 
        #(complex) tfidf representation corresponding to each ngram using 
        #either of the following methods:

        #tfidf.show("bob")

        #tfidf.show_complex("bob", N=10)





    """

    def __init__(self,gramlen=2, verbose = True, spaces = True, normalize = True):
        self.gramlen = gramlen 
        self.spaces = spaces
        self.verbose = verbose
        self.docfreq = Counter()
        self.ndocs = 0
        self.gram_position = {}
        self.normalize = normalize
        self.norm = "l2"
        
        
    def _spaces(self, desc):
        if self.spaces:
            return  " " + desc + " "
        else:
            return desc
        
    def grams(self,desc):
        desc = self._spaces(desc)
        assert len(desc)>=self.gramlen, f"string {desc} is shorter than gramlen {self.gramlen}"
        
        for i in range(len(desc)-self.gramlen+1):
            yield desc[i:i+self.gramlen]
        
    def fit(self,df, desc_col, counts_col=None):
        def update_docfreq(row):
            if counts_col:
                counts = row[counts_col]
            else:
                counts = 1
            desc = self._spaces(row[desc_col])
            

            
            if len(desc)>=self.gramlen:
                
                for gram in self.grams(desc):
                    self.docfreq.update({gram:counts})
                
                self.ndocs += counts
                
        if self.verbose:
            df.progress_apply(update_docfreq,axis=1)
        else:
            df.apply(update_docfreq,axis=1)
            
        #Create map from gram to integers, this will help us in creating the sparse matrices later
        pos = [x[0] for x in self.docfreq.most_common()]
        self.pos = {pos[i]:i for i in range(len(pos))}
        self.nterms = len(self.pos)
            
    def tf(self,desc):
        counter = Counter()
        for gram in self.grams(desc):
            if gram in self.docfreq:
                counter.update({gram:1})
        return counter

    def transform(self,descs):
        if isinstance(descs,str):
            descs =[descs]
        row = []
        col = []
        data = []
        descs = list(enumerate(descs))
        if self.verbose:
            descs = tqdm(descs)
        for i,desc in descs:
            for term, termfreq in self.tf(desc).items():
                docfreq = self.docfreq[term]
                row.append(i)
                col.append(self.pos[term])
                data.append(termfreq*np.log(self.ndocs/docfreq))
                
        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=float)
        
        M = csr_matrix((data, (row, col)), shape=(i+1, self.nterms))
        
        if self.normalize:
            M = normalize(M, norm=self.norm, axis=1)
            
            
        return M
    
    def show(self,desc):
        for term, termfreq in self.tf(desc).items():
            docfreq = self.docfreq[term]
            print((term,termfreq*np.log(self.ndocs/docfreq)))
            

    def complex_tfd(self,desc,N=10):
        dic = dict()
        for i,b in enumerate(self.grams(desc)):
            if b in self.docfreq:
                x, y = dic.get(b,(0,0))
                theta = np.pi*i/(4*N)
                x += np.cos(theta)
                y += np.sin(theta)
                dic[b]=(x,y)
        return dic
    
    def show_complex(self,desc,N=10):
        for term, z in self.complex_tfd(desc,N).items():
            docfreq = self.docfreq[term]
            x,y = z
            idf = (np.log(self.ndocs/docfreq))
            print((term,idf*x, idf *y))
            
    
    def complex_transform(self,descs,N=10):
        if isinstance(descs, str):
            descs = [descs]
        row = []
        col = []
        data = []
        i=0
        for desc in tqdm(descs):   
            
            for b, c in self.complex_tfd(desc,N).items():
                docfreq = self.docfreq[b]
                idf = np.log(self.ndocs/docfreq)
                
                x, y = c
                
                for  k, z in enumerate([x,y]):
                    row.append(i)
                    col.append(self.pos[b] + self.nterms*k)
                    data.append(z*idf)
            i+=1

        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=float)
        M = csr_matrix((data, (row, col)), shape=(len(set(row)), 2*self.nterms))
        
        if self.norm:
            M = normalize(M, norm=self.norm, axis=1)

        return M
            
    def save(self, file):
        f = open(file, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

    def load(self, file):
        f = open(file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        
        
