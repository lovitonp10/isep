
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from scipy.stats import chi2_contingency
    
 
df_stats = pd.read_csv("stats_socio.csv",sep=",")    
print(df_stats.head())

studies = df_stats['Etudes'].value_counts()
parent_cat = df_stats['Parents'].value_counts()


fig,axes = plt.subplots(1,2,figsize=(18,15))
studies.plot(kind='bar',ax=axes[0],title="Etudes")
parent_cat.plot(kind='bar',ax=axes[1],title="Categorie socio-professionnelle")

fig,axes = plt.subplots(1,2,figsize=(18,8))
# Plotting the occurence of each temperature values depending on other attributes
sns.countplot(x="Etudes",data=df_stats,hue='Parents',palette="Blues_d",ax=axes[0]) 
sns.countplot(x="Parents",data=df_stats,hue='Etudes',palette="Blues_d",ax=axes[1]) 


crosstab = pd.crosstab(df_stats['Etudes'],df_stats['Parents'])
print(crosstab.head(6))

def degre_de_liberte(df):
    return((df.shape[0]-1)*(df.shape[1]-1))
    
print("Degrés de liberté : ", degre_de_liberte(crosstab))


chi2, p, dof, ex = chi2_contingency(crosstab)
#df_contagency = pd.DataFrame(ex,index="Ecole_Commerce,Ecole_Ingenieur,IUT/BTS,Univ_Socio,Médecine,Univ_Science".split(","),
#columns="Cadres,Employés,Ouvriers,Prof_Libe".split(','))


def cramers(Index1,Index2,df):
    df_result = pd.crosstab(df[Index1],df[Index2])
    chi2, p, dof, ex = chi2_contingency(df_result)
    n = len(df[Index1])
    r=df_result.shape[0]
    c=len(df_result.columns)
    n=0
    for i in range(0,c):
        for j in range(0, r):
            n=n+df_result[df_result.columns[i]][j]
            
    cramer=(chi2/(n*(min((c-1),(r-1)))))**0.5
    return cramer

print("Chi2 value : ",chi2)
print("Pvalue for the link between the parent background and the type of studies: ",p)
print("Cramer's coefficient :", cramers("Parents","Etudes",df_stats))
