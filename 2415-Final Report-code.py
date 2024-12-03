#!/usr/bin/env python
# coding: utf-8

#  <center><h1>Apple Quantity visualization</h1></center>

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


# In[29]:


get_ipython().system('pip install wordcloud')


# In[32]:


get_ipython().system('pip install graphviz')


# In[37]:


get_ipython().system('brew install graphviz')


# In[2]:


df=pd.read_csv(r"/Users/baby/Desktop/desktop file/2415/Midterm report/apple_quality.csv")


# In[3]:


df.drop(['A_id'],axis=1,inplace=True)
pd.isnull(df).head()


# In[4]:


last_row_index = len(df) - 1
df = df.drop(last_row_index)
pd.isnull(df).sum()


# In[5]:


numeric_data = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_data.corr()

fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale='Blues',
    annotation_text=corr_matrix.round(2).values,
    showscale=True
)

fig.update_layout(title="Correlation Heatmap", width=850, height=850)
fig.show()


# In[17]:


colors = ['#EC9F72', '#8D91C0']  
plt.pie(list(df['Quality'].value_counts()), 
        labels=list(df['Quality'].value_counts().keys()), 
        autopct='%0.1f%%', 
        colors=colors)  
plt.show()


# In[18]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.title('Distribution of Quality')

ax = sns.countplot(x=df['Quality'], palette=['#AEC5D3', '#B3D9CF'])
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()


# In[19]:


df.hist(bins=50, figsize=(20, 15), color='lightblue', edgecolor='black')
plt.show()


# In[20]:


import matplotlib.pyplot as plt
plt.hexbin(df['Size'], df['Sweetness'], gridsize=20, cmap='Blues')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.title('Hexbin Plot of Size vs Sweetness')
plt.show()


# In[25]:


sns.set_style("whitegrid")
sns.jointplot(x="Weight", y="Juiciness", data=df)

plt.show()


# In[26]:


plt.figure(figsize=(8, 6))
plt.scatter(df['Ripeness'], df['Size'], color='blue', alpha=0.5)

plt.xlabel('Ripeness')
plt.ylabel('Size')
plt.title('Scatter Plot: Size vs Ripeness')

plt.grid(True)
plt.show()


# In[6]:


import plotly.express as px

fig = px.scatter(df, x="Sweetness", y="Crunchiness", color="Quality", title='Sweetness x Crunchiness')
fig.update_layout(template='plotly_white')  # Set background to white

fig1 = px.scatter(df, x="Weight", y="Size", color="Quality", title='Weight x Size')
fig1.update_layout(template='plotly_white')  # Set background to white

fig.show()
fig1.show()


# In[26]:


df = pd.read_csv(r"/Users/baby/Desktop/desktop file/2415/Midterm report/apple_quality.csv")

df.drop(['A_id'], axis=1, inplace=True)
last_row_index = len(df) - 1
df = df.drop(last_row_index)

numeric_columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  
df['Quality'] = df['Quality'].map({'good': 'Good Quality', 'bad': 'Poor Quality'})

columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

sns.pairplot(
    data=df,
    vars=columns,
    hue='Quality',
    palette={'Good Quality': 'red', 'Poor Quality': 'green'},
    diag_kind='kde'
)

plt.suptitle("Scatter Plot Matrix of Apple Attributes", y=1.02)
plt.show()


# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"/Users/baby/Desktop/desktop file/2415/Midterm report/apple_quality.csv")
df.drop(['A_id'], axis=1, inplace=True)
last_row_index = len(df) - 1
df = df.drop(last_row_index)

numeric_columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df['Quality'] = df['Quality'].map({'good': 'Good Quality', 'bad': 'Poor Quality'})

plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns):
    plt.subplot(3, 3, i + 1)  
    sns.violinplot(
        data=df,
        x='Quality',
        y=column,
        palette={'Good Quality': 'red', 'Poor Quality': 'green'}
    )
    plt.title(f'Distribution of {column} by Quality')

plt.suptitle("Violin Plots of Apple Attributes by Quality", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


# In[23]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv(r"/Users/baby/Desktop/desktop file/2415/Midterm report/apple_quality.csv")
df = df.dropna(subset=['Quality'])
df['Quality'] = df['Quality'].astype(str)

X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness']]
y = df['Quality']
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X, y)

feature_importances = dict(zip(X.columns, clf.feature_importances_))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='#96B6D8',  
    color_func=lambda *args, **kwargs: "green",  
).generate_from_frequencies(feature_importances)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("Feature Importance Word Cloud", fontsize=16, color="white")
plt.show()


# In[14]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv(r"/Users/baby/Desktop/desktop file/2415/Midterm report/apple_quality.csv")
df = df.dropna(subset=['Quality'])
df['Quality'] = df['Quality'].astype(str)

X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness']]
y = df['Quality']

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(12, 8))  
tree.plot_tree(
    clf, 
    feature_names=list(X.columns),  
    class_names=list(clf.classes_),  
    filled=True, 
    rounded=True, 
    fontsize=8  
)
plt.title("Decision Tree Visualization for Apple Quality")  
plt.show()


# In[ ]:




