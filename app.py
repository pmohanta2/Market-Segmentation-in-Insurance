#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[9]:


loaded_model = pickle.load(open('final_model.sav', 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")


# In[10]:


st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Prediction")


# In[11]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[12]:


with st.form("my_form"):
    balance=st.number_input(label='Balance',format="%.6f")
    balance_frequency=st.number_input(label='Balance Frequency',format="%.6f")
    purchases=st.number_input(label='Purchases',format="%.2f")
    oneoff_purchases=st.number_input(label='OneOff_Purchases',format="%.2f")
    installments_purchases=st.number_input(label='Installments Purchases',format="%.2f")
    cash_advance=st.number_input(label='Cash Advance',format="%.6f")
    purchases_frequency=st.number_input(label='Purchases Frequency',format="%.6f")
    oneoff_purchases_frequency=st.number_input(label='OneOff Purchases Frequency',format="%.6f")
    purchases_installment_frequency=st.number_input(label='Purchases Installments Freqency',format="%.6f")
    cash_advance_frequency=st.number_input(label='Cash Advance Frequency',format="%.6f")
    cash_advance_trx=st.number_input(label='Cash Advance Trx')
    purchases_trx=st.number_input(label='Purchases TRX')
    credit_limit=st.number_input(label='Credit Limit',format="%.1f")
    payments=st.number_input(label='Payments',format="%.6f")
    minimum_payments=st.number_input(label='Minimum Payments',format="%.6f")
    prc_full_payment=st.number_input(label='PRC Full Payment',format="%.6f")
    tenure=st.number_input(label='Tenure')
    
    submitted = st.form_submit_button("Submit")


# In[13]:


data=[[balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,purchases_frequency,oneoff_purchases_frequency,purchases_installment_frequency,cash_advance_frequency,cash_advance_trx,purchases_trx,credit_limit,payments,minimum_payments,prc_full_payment,tenure]]


# In[14]:


if submitted:
    clust=loaded_model.predict(data)[0]
    a=('Data Belongs to Cluster '+str(clust))
    st.success(a)

    cluster_df1=df[df['Cluster']==clust]
    plt.figure(figsize=(5,5))
    for c in cluster_df1.drop(['Cluster'],axis=1):
        grid= sns.FacetGrid(cluster_df1, col='Cluster')
        grid= grid.map(plt.hist, c)
        plt.show()
        st.pyplot()


# In[ ]:




