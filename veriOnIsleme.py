
import pandas as pd
from sklearn.metrics import accuracy_score
from PIL import Image

#NumPy, Python programlama dili için bir kütüphanedir; büyük, çok boyutlu diziler ve matrisler için destek ekler ve bu 
#dizilerde çalışacak geniş bir üst düzey matematiksel işlev koleksiyonu sunar.
import numpy as np
import glob


## veri ölçeklendirilip normalize edildi.
def build_dataset():
	org_dataset = []
	labels = []
    #16 örnek verimiz var.
    #file list dosya listeleme 
    # Glob modülü, Python’da belirli bir klasör içindeki dosyaları listelememize yardımcı olan harika bir modüldür.
	for i in range(1, 16):
		filelist = glob.glob('./data/subject'+str(i).zfill(2)+"*")
		for fname in filelist:
			img = Image.open(fname)
            # image resize küçüt,yeniden boyutlandırma
			img = np.array(img.resize((32, 32), Image.ANTIALIAS))
			img = img.reshape(img.shape[0] * img.shape[1])
            # dönüştürdüğüm image data setine ekle
			org_dataset.append(img)
			labels.append(i)
	return np.array(org_dataset), np.array(labels)

data, labels = build_dataset()

## Normalize
data = data/255
print(len(data))  #165 veri dödürür.




# Test ve Eğitim verisi seti oluşturma
#Her sınıf için 7 görüntü eğitim kalanları test düşünebiliriz.
#her sınıftan 7eğitim 3 test şeklinde oluşturuldu.
#shuffle: True ise, bölmeden önce verileri karıştırır
#random_state, sözde rastgele sayı üretecini kontrol eder. Kodun tekrarlanabilirliği için bir random_state belirtilmelidir.
#stratify : dizi benzeri veya Yok (varsayılan Yoktur)
#Hiçbiri değilse, veriler, bunu labels dizisi olarak kullanarak katmanlara ayrılmış bir şekilde bölünür.
# Bu stratify parametresi, üretilen numunedeki değerlerin oranı, parametre stratify'a sağlanan değerlerin oranıyla aynı olacak şekilde 
# bir bölme yapar.
# Örneğin, y değişkeni 0 ve 1 değerlerine sahip bir ikili kategorik değişkense ve %25 sıfır ve %75 birler varsa, stratify=y rastgele 
# bölmenizin 0'ların %25'ini ve 1'lerin %75'ini olmasını sağlar.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, shuffle=True, random_state=42, stratify=labels)


#Boyutlandırma olmadan sınıflandırıldı.
#sınıflandırma 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy = '  + str(accuracy_score(y_test, y_pred)))



# from sklearn.neighbors import KNeighborsClassifier

# knn=KNeighborsClassifier(n_neighbors=4)
# knn.fit(X_train,y_train)

# knn.score(X_test,y_test)

# result=knn.predict(X_test)

# print(pd.crosstab(y_test,result,rownames=['Real'],colnames=['Predicted'],margins=True, margins_name='total'))


# k_list = list(range(1,25))



# k_values = dict(n_neighbors=k_list)
# print(k_values.keys()),
# print(k_values.values())






from sklearn.model_selection import train_test_split
import math

# #2. Benzerlik hesaplama
def dist(x1, x2):
    total = 0
    for i, j in zip(x1, x2):
        total += math.pow(i - j, 2)

    return math.sqrt(total)

# # En yakın komşuları bulma
def get_nearest_neighbors(row0, x_train, k):
    distances = []
    neighbors = []
    for i, row1 in enumerate(x_train):
        c = dist(row0, row1)
        distances.append([c, i])
        
    distances.sort(key = lambda x: x[0])
    for j in range(k):
          neighbors.append(distances[j])

    return neighbors



def KNN(K, X_test, X_train, y_train):
    
    Y_predict = []

    for x_test in X_test:
        neighbors = get_nearest_neighbors(x_test, X_train, K)
        targets = []
        for n in neighbors:
            index = n[1]
            targets.append(y_train[index])

        Y_predict.append(max(targets, key = targets.count))

    return Y_predict


y_pred = KNN(1, X_test, X_train, y_train)


from sklearn.metrics import r2_score
# #r2_score fonksiyonu,makine öğrenmesi algoritmalarının başarılarını ölçen genel bir fonksiyondur. 
# #Tahminlerle gerçek değerlerin ne kadar uyuştuğunu hesaplayıp yüzde olarak değer döndürür
print(r2_score(y_pred, y_test))


# #   #LLE
from sklearn.manifold import LocallyLinearEmbedding
embedding = LocallyLinearEmbedding(n_neighbors = 79, n_components=108)
trainKlpp = embedding.fit_transform(X_train)
testKlpp = embedding.transform(X_test)



# import LE
# embedding =LE(n_components=2)
# X_transformed = embedding.fit_transform(X_train)
# X_Testing=embedding.transform(X_test)



# from KLPP import constructWKLPP, KLPP, constructKernelKLPP
# gnd = y_train #98
# options={}
# options['NeighborMode'] = 'Supervised'
# options['gnd'] = gnd
# options['WeightMode'] = 'HeatKernel'
# options['bLDA'] = 1
# options['bNormalized'] = 1
# options['t'] = 20
# options['reducedDim'] = 35
# options['k'] = 125
# W = constructWKLPP(X_train, options)
# options['KernelType'] = 'Gaussian'
# options['Regu'] = 1
# options['ReguAlpha'] = 0.001
# eigvector, eigvalue = KLPP(W, options, X_train)
# kTrain = constructKernelKLPP(X_train, [], options)
# trainKlpp = np.dot(kTrain,eigvector)
# kTest = constructKernelKLPP(X_test, X_train, options)
# testKlpp = np.dot(kTest,eigvector)


















# from sklearn.manifold import LocallyLinearEmbedding
# for i in range(10):
#     for j in range(20):
      
#         print(str((i+1)*3) + str((j+1)*5))
#         embedding = LocallyLinearEmbedding(n_neighbors = (i+1)*3, n_components=(j+1)*5)
#         trainKlpp = embedding.fit_transform(X_train)
#         testKlpp = embedding.transform(X_test)
    

y_pred = KNN(1, X_test, X_train, y_train)
print(r2_score(y_pred, y_test))



def KNN(K, testKlpp, trainKlpp, y_train):
    
    Y_predict = []

    for x_test in testKlpp:
        neighbors = get_nearest_neighbors(x_test,trainKlpp, K)
        targets = []
        for n in neighbors:
            index = n[1]
            targets.append(y_train[index])

        Y_predict.append(max(targets, key = targets.count))

    return Y_predict


y_pred = KNN(1, testKlpp, trainKlpp, y_train)


from sklearn.metrics import r2_score
#r2_score fonksiyonu,makine öğrenmesi algoritmalarının başarılarını ölçen genel bir fonksiyondur. 
#Tahminlerle gerçek değerlerin ne kadar uyuştuğunu hesaplayıp yüzde olarak değer döndürür
print('accuracy with lle' )
print(r2_score(y_pred, y_test))


# import lle

# ks = list(range(1,15))
# knn_ks = list(range(1,15))


# for a in ks:
# 	for knn_k in knn_ks: 
# 		lleModel = lle( k = a )
# 		selfObject = lleModel.fit(X_train)
# 		trainKlpp = np.dot(X_train, selfObject.projection_)
# 		testKlpp = np.dot(X_test, selfObject.projection_)

# 			## classification
# 			# default=2 Minkowski metric
# 		from sklearn.neighbors import KNeighborsClassifier
# 		model = KNeighborsClassifier(n_neighbors = knn_k)
# 		model.fit(trainKlpp, y_train)
# 		y_pred = model.predict(testKlpp)
# 		print('Accuracy with LPP = '  + str(accuracy_score(y_test, y_pred)))


# model =KNN(1, X_test, X_train, y_train)
# model.fit(trainKlpp, y_train)
# y_pred = model.predict(testKlpp)
# print('Accuracy with KLPP = '  + str(accuracy_score(y_test, y_pred)))













