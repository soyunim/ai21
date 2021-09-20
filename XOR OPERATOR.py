#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import random
from math import exp,log


# In[27]:


X=np.array([[0,0],[0,1],[1,0],[1,1]]) 
Y=np.array([[0],[1],[1],[0]])


# In[28]:


class logistic_regression_model():
    def __init__(self):
        self.w=np.random.normal(size=2)
        self.b=np.random.normal()
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
                  
    def predict(self,x):
        z=np.inner(self.w,x)+self.b
        a=self.sigmoid(z)
        return a


# In[29]:


model = logistic_regression_model()


# In[30]:


def train(X, Y, model, lr):
    dw0=0.0
    dw1=0.0
    db=0.0
    m=len(X)
    cost=0.0
    for x,y in zip(X,Y):
        a=model.predict(x)
        if y==1:
            cost -= log(a)
        else:
            cost -= log(1-a)
            
        this=np.append(x,1)
        [dw0,dw1,db] = [dw0,dw1,db] + ((a-y)*this)
        
    cost /= m
    model.w[0] -= lr*dw0/m
    model.w[1] -= lr*dw1/m
    model.b -= lr*db/m
    
    return cost


# In[31]:


def loss():
    losses = []
    for i in range(4):
        loss = -Y[i]*np.log(model.predict(X[i]))-(1-Y[i])*np.log(1-model.predict(X[i]))
        losses.append(loss)
    return losses


# In[32]:


for epoch in range(10000):
    cost = train(X,Y,model,0.1)
    if epoch%100==0:
        print(epoch, cost)


# In[33]:


model.predict((0,0))


# In[34]:


model.predict((0,1))


# In[35]:


model.predict((1,0))


# In[36]:


model.predict((1,1))


# In[37]:


loss_01 = loss()
loss_01


# In[38]:


for epoch in range(10000):
    cost = train(X,Y,model,0.01)
    if epoch%100==0:
        print(epoch, cost)


# In[39]:


model.predict((0,0))


# In[40]:


model.predict((0,1))


# In[41]:


model.predict((1,0))


# In[42]:


model.predict((1,1))


# In[43]:


loss_001 = loss()
loss_001


# In[44]:


for epoch in range(10000):
    cost = train(X,Y,model,1.0)
    if epoch%100==0:
        print(epoch, cost)


# In[45]:


model.predict((0,0))


# In[46]:


model.predict((0,1))


# In[47]:


model.predict((1,0))


# In[48]:


model.predict((1,1))


# In[49]:


loss_10 = loss()
loss_10


# In[50]:


plt.title('XOR OPERATOR')
plt.ylabel("loss")
arr_x = ["(0,0)","(0,1)","(1,0)","(1,1)"]
plt.plot(arr_x,loss_01,'c',label='0.1')
plt.plot(arr_x,loss_001,'m',label='0.01')
plt.plot(arr_x,loss_10,'y',label='1.0')
plt.legend(loc="upper right")
plt.show()

