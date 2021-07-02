import Models
import keras
import numpy as np
import pandas as pd

# df = pd.read_csv("/home/amithbn/Desktop/TABL/fi_2010_dataset/FI2010_train.csv")
# arr = df["148"].values


# n = len(df)

# n /= 10
# n = int(n)
# arr = arr[0:n]


# up = pd.DataFrame(data={"UP":np.where(arr == 1, 1, 0)})
# dn = pd.DataFrame(data={"DN":np.where(arr == 3, 1, 0)})
# #eq = pd.DataFrame(data={"EQ":np.where(arr == 2, 1, 0)})

# df = df.iloc[:, :40]
# lst = np.array_split(df,n)
# lst = np.array(lst)


# lbls = up.join(dn)
# #lbls = lbls.join(eq)



# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
# template = [[10,40], [120,5], [2,1]]
template = [[40,10], [120,5], [3,1]]

x = np.random.rand(1000,40,10)
y = keras.utils.to_categorical(np.random.randint(0,3,(1000,)),3)

# random data
#y = keras.utils.to_categorical(np.random.randint(0,3,(n,)),3) # - low, high, 

# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0,axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)
dropout = 0.1

 
model = Models.TABL(template, dropout, projection_regularizer, projection_constraint,
                    attention_regularizer, attention_constraint)
model.summary()

# create class weight
class_weight = {0 : 1e6/300.0,
                1 : 1e6/400.0,
                2 : 1e6/300.0}


# training
model.fit(x,y, batch_size=256, epochs=100, class_weight=class_weight)


# model.fit(lst,lbls, batch_size=256, epochs=10, class_weight=class_weight)
# model.save('tabl_model.h5')

