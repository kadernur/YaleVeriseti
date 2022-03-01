import numpy

def LLE(M, k, m, quiet=False):
    M = numpy.matrix(M)
    # print(M)
    d,N = M.shape
    assert k<N

    #ağırlık matrisini oluştur
    W = numpy.zeros((N,N))
    m_estimate = []
    var_total = 0.0
    
    for row in range(N):
        #k en yakın komşuyu bul
        M_Mi = numpy.array(M-M[:,row])
    
        vec = (M_Mi**2).sum(0)
        nbrs = numpy.argsort(vec)[1:k+1]
        
      
        #komşulara göre ağırlık vektörünü hesapla
      
       #mesafelerin kovaryans matrisini hesapla
        M_Mi = numpy.matrix(M_Mi[:,nbrs])
        Q = M_Mi.T * M_Mi
        
        #M_Mi'nin tekil değerleri varyansı verir, bunu içsel boyutluluğu hesaplamak için kullanın
        
        sig2 = (numpy.linalg.svd(M_Mi,compute_uv=0))**2
        #komşuluktaki içsel boyutluluğu hesaplamak için 
        #sig2 yi kullanırız.
        
        v=0.9
        sig2 /= sig2.sum()
        S = sig2.cumsum()
        m_est = S.searchsorted(v)
        if m_est>0:
            m_est += ( (v-S[m_est-1])/sig2[m_est] )
        else:
            m_est = v/sig2[m_est]
        m_estimate.append(m_est)
        
        
        #Kovaryans matrisi neredeyse tekil olabilir: 
       #sayısal hataları önlemek için bir diyagonal düzeltme ekleyin. 
       #düzeltme (d-m) kullanılmayan varyansların toplamına eşittir
       #izle karşılaştırıldığında küçük bir düzeltme
       #r = 0.001 * float(Q.trace())
        r = numpy.sum(sig2[m:])
        var_total += r
        Q.flat[::k+1] += r
        
        #ağırlık matrisini çözme.
        w = numpy.linalg.solve(Q,numpy.ones(Q.shape[0]))
        w /= numpy.sum(w)
        W[row,nbrs] = w


    #sıfır uzayını bulmak için, (W-I).T*(W-I)'nin alt d+1 özvektörlerine ihtiyacımız var. 
    #Bunu (W-I)'nin svd'sini kullanarak hesaplanır.
    I = numpy.identity(W.shape[0])
    U,sig,VT = numpy.linalg.svd(W-I,full_matrices=0)
    indices = numpy.argsort(sig)[1:m+1]
    
    return numpy.array(VT[indices,:])



from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import glob

def build_dataset():
	org_dataset = []
	labels = []
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
data = data/255
#print(len(data))  #165 veri dödürür.


from sklearn.model_selection import train_test_split
import math
def dist(x1, x2):
    total = 0
    for i, j in zip(x1, x2):
        total += math.pow(i - j, 2)
    return math.sqrt(total)

def get_nearest_neighbors(row0, x_train, k):
    distances = []
    neighbors = []
    for i, row1 in enumerate(x_train):
        c = dist(row0, row1)
        distances.append([c, i])
    distances.sort(key = lambda x: x[0])  # uzaklığa göre sırala
    for j in range(k):
         neighbors.append(distances[j])
    return neighbors


def KNN(K, X_test, X_train, Y_train):
    Y_predict = []
    for x_test in X_test:
        neighbors = get_nearest_neighbors(x_test, X_train, K)
        targets = []
        for n in neighbors:
            index = n[1] #i
            targets.append(Y_train[index])
        Y_predict.append(max(targets, key = targets.count))#en fazla tekrar eden sınıfı döndürür
    return Y_predict


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, shuffle=True, random_state=42, stratify=labels)

y_pred = KNN(1, X_test, X_train, y_train)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
print('Accuracy  = '  + str(accuracy_score(y_test, y_pred)))


from sklearn.metrics import r2_score

best_acc, best_k = 0.0, -1

k_vals, acc_vals, r_vals = [], [], []

for k in range(1, 10):
   
    vt = LLE(X_train, k, 110) ## donusum matrisini verir
    X_trainn =  np.dot(X_train, np.transpose(vt))
    X_testn =  np.dot(X_test, np.transpose(vt))
    y_pred = KNN(1, X_testn, X_trainn, y_train)
    
    acc = accuracy_score(y_test, y_pred)
    k_vals.append(k)
    acc_vals.append(acc)
    r_vals.append(r2_score(y_test, y_pred))
    
    if acc > best_acc:
        best_acc = acc
        best_k = k
    
    #print(r2_score(y_test, y_pred))
    #print('Accuracy  With LE = '  + str(accuracy_score(y_test, y_pred)))
    print(f"Accuracy With LLE k = {k}, acc = {acc:.4f}")
    
print(f"Best Accuracy With LLE k = {best_k}, acc = {best_acc:.4f}")

import matplotlib.pyplot as plt

plt.title('LLE basarı grafigi')
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.plot(k_vals, acc_vals)
plt.plot(k_vals, r_vals)
plt.legend(['accuracy', 'r2 score'])
plt.grid()
plt.show()
    

# vt = LLE(X_train, 7, 80) ## donusum matrisini verir
# X_trainn =  np.dot(X_train, np.transpose(vt))
# X_testn =  np.dot(X_test, np.transpose(vt))
# y_pred = KNN(1, X_testn, X_trainn, y_train)
# from sklearn.metrics import r2_score
# print(r2_score(y_test, y_pred))
# print('Accuracy  With LE = '  + str(accuracy_score(y_test, y_pred)))

