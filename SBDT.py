import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
import os

class UI():
    def __init__(self):
        self._tree_model = SBDT()
        self._trained = False
        self._options = ("train" , "test", "predict" , "stop")
        self.train_shape = tuple()
        self.test_shape = tuple()

    def start(self):
        print("INFO")
        print("Hello , this is SBDT model for binary classification")
        print("You can train, test model and predict new observatoin by trained model")
        print("Data should be in numpy format")
        print("Train and test data should be with script in one folder")
        print("Train data format: \n M_TRxN numerical matrix , where last column is a target variable and others are input variables ")
        print("Test data format: \n M_TESTxN matrix with same number of culumns train data  ")
        print("Input commnad format : \ncommand --filename")
        print("commands : {0}".format(self._options))
        print("-------------------------------------")
        self._test_shape = tuple()
        self._train_shape = tuple()
        
    def mainLoop(self):
        while True:
            data = input("Choose one of the options {0} ".format(self._options)).replace(" ",'')
            option = data.split("--")[0]
            if option not in self._options:
                print(" Invalid option " , option)
            else:
                if option == "stop":
                    break
                file = data.split("--")[-1]
                print("File" , file)
                if len(file) < 2 or '' in set(file) or " " in set(file) or file.split(".")[-1] != "npy" or "--" not in data:
                    print("invalid file name")
                    continue
                if option == "train":
                    self._train(file)
                if option == "test":
                    self._test(file)
                if option == "predict":
                    self._predict(file) 
                
        self._stop()

    def _train(self,file):
        print("TRAIN")
        if file not in os.listdir():
            print("File not in dir")
            return
        data = np.load(file)
        X = data[:,:-1]
        y = data[:,-1]
        self._train_shape = data.shape
        self._tree_model.fit(X,y)
        self._trained = True
        acc = self._tree_model.evaluete(X,y)
        print("Accuracy for train {0}".format(acc))
    def _test(self,file):
        print("TEST")
        acc = -1
        if file not in os.listdir():
            print("File not in dir")
            return
        if self._trained:
            data = np.load(file)
            self._shape = data.shape
            X = data[: , :-1]
            y = data[: , -1]
            acc = self._tree_model.evaluete(X,y)
            print("Accuracy for test {0}".format(acc))
        else:
            print("Train model in first")

    def _predict(self,file):
        print("PREDICT")
        if file not in os.listdir():
            print("File not in dir")
            return
        data = np.load(file)
        data_shape = data.shape
        if len(data_shape) != 2:
            print("invalid dimention of input data , Need 2-dimentional , got {0}".format(len(data_shape)))
        else:
            data_vector_size = data_shape[1]
            if data_vector_size != self._train_shape[1] - 1:
                print("invalid size of predict vector, got {0}".format(data_vector_size))
            else:
                preds = self._tree_model.predict(data)
                print("predictions {0}".format(preds))
        return
    def _stop(self):
        print("stop")
        quit()
    
class SBDT():
    def __init__(self , max_depth = 3 , mode ='sbdt'):
        self.max_depth = max_depth
        self.mode = mode
        
    def generate_support_points(self , data , labels , num_points = 100 , subsample_size = 0.6 , num_clusters = 2):
        feature_size = data.shape[1]
        point_matrix = np.zeros((num_points,num_clusters,feature_size))
        sub_size = int(data.shape[0] * subsample_size)
        data_pos = data[labels == 1]
        data_neg = data[labels == 0]
        print('pos shape {0} , neg_shape {1}'.format(data_pos.shape , data_neg.shape))
        for i in range(num_points):
            sub_data_pos = data_pos[np.random.randint(0 , data_pos.shape[0] , size = int(sub_size // 2))]
            sub_data_neg = data_neg[np.random.randint(0 , data_neg.shape[0] , size = int(sub_size // 2))]
            sub_data = np.concatenate((sub_data_pos , sub_data_neg), axis = 0)
            kmeans = KMeans(n_clusters = 2, random_state = 0).fit(sub_data)
            centers = kmeans.cluster_centers_
            point_matrix[i,:,:] = centers
        return point_matrix
    def transform_to_new_features(self,  data , sup_points , dist_matrix = None):
        num_pair_points = sup_points.shape[0]
        dist_relative_matrix = np.zeros((data.shape[0] , num_pair_points , num_pair_points) , dtype = 'uint8')
        dist1 = cdist(data , sup_points[:,0,:])
        dist2 = cdist(data , sup_points[:,1,:])
        for d in range(data.shape[0]):
            for i in range(num_pair_points):
                for j in range(num_pair_points):
                    if dist1[d,i] > dist2[d,j]: dist_relative_matrix[d,i,j] = 1
                    else: dist_relative_matrix[d,i,j] = 0
        return dist_relative_matrix.reshape((data.shape[0] , num_pair_points**2))
    def build_tree(self , data , labels , max_depth):
        clf = DecisionTreeClassifier(random_state=0,max_depth = max_depth)
        clf = clf.fit(data,labels)
        self.tree = clf
    def evaluete(self , X , Y):
        X = self.transform_to_new_features(X , self.points)
        preds = self.tree.predict(X)
        return ((Y == preds).sum()) / (Y.shape[0])
    def predict(self , X):
        X = self.transform_to_new_features(X , self.points)
        return self.tree.predict(X)
    def fit(self , X , y):
        if self.mode == 'classic':
            clf = DecisionTreeClassifier(random_state = 0 , max_depth = self.max_depth)
            clf = clf.fit(X , y)
            self.tree = clf
            return self
        else:
            self.points = self.generate_support_points(X , y , num_points = 100 , subsample_size = 0.2)
            features = self.transform_to_new_features(X , self.points)
            self.build_tree(features , y , max_depth = self.max_depth)
            return self

def main():
    ui = UI()
    ui.start()
    ui.mainLoop()
main()