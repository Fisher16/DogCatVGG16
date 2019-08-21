import shutil
train_n=1000
for i in range(train_n):
    shutil.copy(f'dc/train/cat.{i}.jpg',f'dc/train_1k/cats/{i}.jpg')
    shutil.copy(f'dc/train/dog.{i}.jpg',f'dc/train_1k/dogs/{i}.jpg')
val_n=250
for i in range(train_n+1,train_n+val_n+1):
    shutil.copy(f'dc/train/cat.{i}.jpg',f'dc/val/cats/{i}.jpg')
    shutil.copy(f'dc/train/dog.{i}.jpg',f'dc/val/dogs/{i}.jpg')
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)
model.summary()

test=image.load_img('dc/train/cat.0.jpg', target_size=(224, 224))
image.save_img('dc/prez/after_scale.jpg', test)

def extract_features(path):
    #VGG16 trained on images of size 224x224
    img = image.load_img(path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    return vgg16_feature.flatten()
arr_train=[]
path='dc/train_1k/{}/{}.jpg'
for i in range(train_n):
    if(i%100==0):
        print(i/100)
    arr_train.append(np.concatenate(([1],extract_features(path.format('dogs',str(i))))))
    arr_train.append(np.concatenate(([0],extract_features(path.format('cats',str(i))))))
import pandas as pd
df=pd.DataFrame(np.array(arr_train))
#extracted features first col target
#df.to_csv('dog_cats.vgg16.features.csv')
from sklearn.preprocessing import StandardScaler
# Separating out the features
x = df.loc[:, 1:].values
# Separating out the target
y = df.loc[:,0].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','pc3'])
finalDf = pd.concat([principalDf, df[0]], axis = 1)
finalDf.columns=['pc1','pc2','pc3','target']
finalDf.to_csv('dog_cats.1k.vgg16.features.pca.csv')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111,projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
colors = ['r', 'b']
targets = [1,0]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'],
               finalDf.loc[indicesToKeep, 'pc2'],
               finalDf.loc[indicesToKeep, 'pc3'],
               c = color,
               s = 25,
               alpha=0.25
                )
ax.legend(targets,labels=['Dog','Cat'])
ax.grid()
ds_train=finalDf

arr_val=[]
path='dc/val/{}/{}.jpg'
for i in range(train_n+1,train_n+val_n+1):
    if(i%50==0):
        print(i/50)
    arr_val.append(np.concatenate(([1],extract_features(path.format('dogs',str(i))))))
    arr_val.append(np.concatenate(([0],extract_features(path.format('cats',str(i))))))
df_val=pd.DataFrame(np.array(arr_val))
x_val = df_val.loc[:, 1:].values
y_val = df_val.loc[:,0].values
x_val = StandardScaler().fit_transform(x_val)
pca_val = pd.DataFrame(data = pca.transform(x_val), columns = ['pc1', 'pc2','pc3'])
ds_val = pd.concat([pca_val, df_val[0]], axis = 1)
ds_val.columns=['pc1','pc2','pc3','target']
ds_val.to_csv('dog_cats.1k.vgg16.features.pca.val.csv')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
nb_model = gnb.fit(ds_train.iloc[:,:-1], ds_train.iloc[:,-1])
y_pred=nb_model.predict(ds_val.iloc[:,:-1])
n=ds_val.iloc[:,:-1].shape[0]
acc=(ds_val.iloc[:,-1]== y_pred).sum()/n
print("Accuracy: {}%".format(acc*100))
miss=ds_val[ds_val.iloc[:,-1]!= y_pred].index
for i in miss:
    if(i%2==0):
        shutil.copy('dc/val/dogs/{}.jpg'.format(int(1000+i/2)),'dc/val/miss_nb')
    else:
        shutil.copy('dc/val/cats/{}.jpg'.format(int(1000+i/2)),'dc/val/miss_nb')
from sklearn import svm
svm_model = svm.SVC(gamma='scale')
svm_model.fit(ds_train.iloc[:,:-1], ds_train.iloc[:,-1])
y_pred=svm_model.predict(ds_val.iloc[:,:-1])
acc=(ds_val.iloc[:,-1]== y_pred).sum()/n
print("Accuracy: {}%".format(acc*100))
miss=ds_val[ds_val.iloc[:,-1]!= y_pred].index
for i in miss[1:]:
    if(i%2==0):
        shutil.copy('dc/val/dogs/{}.jpg'.format(int(1000+i/2)),'dc/val/miss_svm')
    else:
        shutil.copy('dc/val/cats/{}.jpg'.format(int(1000+i/2)),'dc/val/miss_svm')