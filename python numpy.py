# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

obje = pd.Series(
   ["ali","mert","hasan",20],
   index=["a","b","c","d"]
    )

#print(obje[2])
#print(obje)
#print(obje["d"])

puan = {
        "ali":90,
        "mert":75,
        "samet":59}

nt = pd.Series(puan)


print(nt[nt>60])

nt["ali"] = 25
print(nt)


nt**2

nt.isnull()

isimler = pd.Series(
    ["harun","samet","kaan","mert","cenk"],
    index = ["a","b","c","d","e"]
    )

notlar = pd.Series(
     [10,58,63,54,47]
     )

print(notlar[[2,3]])



kisiler = pd.Series(
    ["ahmet","samet","kerem","mert","cenk"]
    )

d= {
    "kisiler":kisiler,
    "notlar":notlar
    }

dataframe = pd.DataFrame(d)

print(pd.DataFrame(d).head(2))


dataframe.shape
dataframe.columns
dataframe["kisiler"]
dataframe["notlar"]
dataframe[["notlar","kisiler"]]




pd.Series(
    ["MERHABA","BENİM","ADIM","HARUN"]
    )[[0,2]]


print(isimler)
print(isimler["a"])

df = pd.read_excel (r'C:\Users\Harun\Desktop\R dataset\MarketSales.xlsx')



pd.DataFrame(df.columns)


df.head(5)[["GENDER","BRAND"]]
df["ITEMNAME"][[1,2,3,4,5,6]]
df.shape
df.dtypes
pd.Series(
    df.columns
    )

df.ITEMNAME.describe()
df.ITEMNAME.value_counts()



isim = pd.Series(
    ["merhaba","ben","harun"]
    )

notlar = pd.Series(
    [10,20,30]
    )

Tablo = {
    "isim":isim,
    "notlar":notlar
    }

pd.DataFrame(
    Tablo).notlar

np.array(
    ["harun",10,20]
    )


dataframe.shape

baslik = pd.Series(
    dataframe.columns,
    index=["a","b"]
    )


import numpy as np

#numpy arraylerinde bu değişkenlerin bilgilerini tek tek tutmaz 

a=[1,2,3,4] #bu vektör
b=[2,3,4,5] # bu listede 4 tane int var bunu saklamanın maliyeti var

#fakat bu numpy arrayi 4 kat daha az maliyetli her bir eleman için tuttuğu bilgiyi
#sadece bu array için tutuyor bu int dır.

a=np.array(
    [1,2,3,4]
    )

b=np.array(
    [2,3,4,5]
    )

a*b 
sum(a*b)


# on tane 0 oluşturdu ondalıklı idi int yaptı.
np.zeros(
    10,
    dtype=int
    )

np.ones((3,5) , dtype=int)

np.full((3,5),12,dtype=int)


np.arange(0,31,4)   #sık sık kullancagım
len(np.arange(0,31,4))  #0 dan 31 'e kadar 4'er 4'er artsın

np.linspace(0,1,10) # 0-1 arası 10 tane sayı luturdu


np.array(
    np.arange(1,50,5)
    ).reshape((2,5))


np.random.normal(10,2,15)  
np.random.normal(10,2,(3,5))  #sık sık kullancagım

np.random.randint(0,10,(3,5)) # ,-10 arası rastgele sayı üret.

np.random.randint(50,100,5)  #sık sık kullancagım



dataframe.size

dataframe.notlar[np.arange(0,3)]

a = np.random.randint(10,size=10)

a.ndim
a.shape
a.size   #toplam eleman sayısı.
a.dtype


b=np.random.randint(10,size=(3,5))
b
b.ndim
b.shape
b.size
b.dtype

np.arange(1,10).reshape((3,3))

x=np.array([1,2,3])
y=np.array([4,5,6])
np.concatenate([x,y])  #iki vektörü yanyana dizdi cbind fonksiyonu gibi

z=np.array([7,8,9])

np.concatenate([x,y,z])  

#cbind fonksiyonu.

a = np.array([[1,2,3],
          [4,5,6],]
         )

np.concatenate([a,a],axis=1) #axis = eksen demek 1 dersen yanyana

#axis =0 dersen altalta birleştirir.

np.concatenate([a,a],axis=0) #altalta birleştirdi.


x=np.array([1,2,3,99,99,3,2,1])


np.split(x,[3,5]) # 3'e kadar böl sonra 5'e kadar böl gerisini yazmaya gerek yok.

splitVektor = np.split(x,[3,5])

np.concatenate([splitVektor[0],
                splitVektor[1],
                splitVektor[2]]) #tekrar birleştirdi.

#bu 3 değişkene çıktıyı atadı.
a,b,c =  np.split(x,[3,5])

m=np.arange(16).reshape((4,4))

np.vsplit(m,[2])

ust , alt = np.vsplit(m,[2])


x[x>10]
dataframe[dataframe.notlar>20]



#PANDAS KÜTÜPHANESİ

import pandas as pd

#numpy arraylerinden farkı burada index bilgisi vardır.

seri  = pd.Series([20,45,3,4,5])

type(seri)

seri.axes #index bilgileri 
seri.dtypes
seri.size
seri.ndim
seri.shape
seri.values #indeksler olmadan sanki arraymiş gibi aldım.
dataframe.values
seri.head(2)
seri.tail(2)

seri = pd.Series([1,2,3,4,5],index=["a","b","c","d","e"])
seri["b"]
seri["a":"c"]
seri[["a","b"]]


sozluk = pd.Series({"log":10,
          "reg":20})


pd.concat(
    [sozluk,sozluk]
    )

pd.concat(
    [sozluk,sozluk],
    axis=1
    )




dataframe[["kisiler","notlar"]]

pd.concat([dataframe.notlar,dataframe.kisiler])
pd.concat([dataframe.notlar,dataframe.kisiler],axis=1)

np.arange(0,10,2)


dataframe.columns = pd.Series(["Kisi","Not"])


# Ders 131

import numpy as np
a=np.array([1,2,33,444,75])
seri = pd.Series(a)
seri
seri[0]
seri[0:2]

dataframe[0:3][["Kisi","Not"]]

dataframe[2:4][0:1]

seri = pd.Series([121,200,150,99],
                    index=["reg","log","cart","rf"])


seri.index
seri.keys()
list(seri.items())

seri.values

seri[seri>150]

#eleman sorgulama

"reg" in seri.keys()
"a" in seri.keys()


#fency eleman

seri[["rf","log"]]
seri[["reg"]]=130

seri["reg"]


seri.index
seri["reg":"cart"]

# ders 132 ********************************************************************
import pandas as pd
import numpy as np

df = pd.read_excel (r'C:\Users\Harun\Desktop\R dataset\MarketSales.xlsx')

a=df

pd.Series(a.columns)

a[a.SALESMAN==][["SALESMAN","CITY"]]

a[1:5][["SALESMAN","CITY","BRANCH"]]



m=np.array([1,2,3,4,5,6,7,8,9])


m=m.reshape([3,3])

m=pd.DataFrame(m)

m.columns=["var1","var2","var3"]


m.head()

m[1:2][["var1","var2"]]

df=m


type(df)


df.axes
df.shape
df.ndim
df.size
df.shape[1]
df.values
df.head()
df.tail()

df.iloc[1:2,1:2]


#Ders 133 *************************************************************

df.iloc[1:5][["STARTDATE","ITEMNAME"]]


s1=np.random.randint(10,size=5)
s2=np.random.randint(10,size=5)
s3=np.random.randint(10,size=5)

sozluk = {
    "var1":s1,
    "var2":s2,
    "var3":s3
    }

sozluk["var1"]

data = pd.DataFrame(sozluk)


data.iloc[1:2][["var1","var2"]]

#silme

data
data.drop(1,axis=0) #axis 0 ise o satırı siler 1 isesutunu siler.

df.info()

df.describe().T
df.isnull()
df.corr()
data.corr()

data.index = ["a","b","c","d","e"]
data

data.drop("a",axis=0) # a satırını sildi.

data

l=["c","e"]
data.drop(l,axis=0)

data.isnull()

# degiskenler için

degisken = input("Degisşken adı giriniz :")

sonuc = degisken in data

if sonuc == True:
    print("evet var")
else:
    print("hayır yok")
    

l=["var1","var2","1212"]


for i in l:
    print(i in data)


data["var4"] = data["var1"] * data["var2"]


data.drop("var4",axis=1) #sutun sildim.

data

l=["var1","var2"]

data.drop(l,axis=1)

#Ders 134 ----------------------------------------------------------
#loc ve iloc

m=np.random.randint(1,30,size=(10,3))

df = pd.DataFrame(m,columns = ["var1","var2","var3"])
df

df[1:10][["var2","var1"]]

df.loc[0:3][["var1","var2"]]

df.iloc[:3,:2]

df.loc[df.var1>15 ,["var1","var2","var3"]]


df.loc[1:5,["var1","var2"]]

df.loc[df.var2>20,["var1","var3"]]

df.loc[df.var1>15 ,]

#coklu eleman seçimi
df.loc[(df.var1>10) & (df.var3>5) , ["var1","var3"]]


df.loc[
       (df.var2>np.average(df.var1)) &
       (df.var2>np.average(df.var2)),
       ["var1","var2"]
       ]


#Ders 136 -----------------------------------------

df.loc[(df.ITEMNAME == "FALIM SAKIZ 5LI NANE") & (df.CITY=="İstanbul"),
       ["CITY","ITEMNAME"]]


df1 = np.random.randint(1,30,size = (5,3))

df1 = pd.DataFrame(df1,columns=["var1","var2","var3"])
df2 = df1 + 99

#altalta birleştirdi.
df3 = pd.concat([df1,df2],ignore_index=1)

df3.loc[
       (df3.var1>np.average(df3.var1)) & (df3.var2>np.average(df3.var2)),
       ["var1","var3","var2"]
       ]


df2.columns = ["var1","var2","deg3"]
df2
pd.concat([df1,df2]) #degişken isimleri aynı olmadığı için birleştirmedi.
  

pd.concat([df1,df2],join="inner",ignore_index = True) #sadece aynı bilgiye sahip 2 kolonu birleştirdi.


#Ders 138 ---------------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns #veri seti yüklemek için.

df.groupby(by=df.CITY).count().loc[:,"ITEMNAME"]

a=np.arange(1,10).reshape((3,3))
a=pd.DataFrame(a)


nall = np.random.randint(0,6,2) # sayı üretir

a.columns=["var1","var2","var3"]

a=pd.concat([a,a] ,ignore_index=True)

# iki kolona nan deger atadım.
a.loc[nall,["var1"]]=np.nan
a.loc[nall,["var2"]]=np.nan


#eksik olanı ortalamayla doldurdum.
a["var1"]=a["var1"].fillna(np.nanmean(a["var1"]))


a["var2"]=a["var2"].fillna(np.nanmean(a["var2"]))


df = sns.load_dataset("planets")
df.shape
df.head()
df.columns
df.mean()
df.count()
df["mass"].mean()
df.loc[: , "mass"].count()
df["mass"].std()
df["mass"].var()
df["mass"].min()
df.max()
df.var()
df.describe().T

df.dropna().describe().T # eksik veri varsa bunu kullan
df["orbital_period"].describe().T


a=[]
for i in range(1,len(df["orbital_period"].isna())):
    if df["orbital_period"].isna()[i]==True:
        a.append(i)
        
df.loc[a,:]


df.groupby(df["method"]).count()


# 10 ar 10 ar git.
np.arange(1,30,10) 



#ders 139------------------------------------------------------------------



df=pd.DataFrame(
    {"gruplar":["A","B","C","A","B","C"],
     "veri":[10,11,52,23,43,55]
     },
    columns = ["gruplar","veri"]
    )


df

df.groupby("gruplar").mean()
df.groupby("gruplar").count()
df.groupby("gruplar").min()
df.groupby("gruplar").max()
df.groupby("gruplar").std()
df.groupby("gruplar").var()

df.groupby("gruplar").describe()

df = sns.load_dataset("planets")


df.groupby("method").describe()


#orbitale göre ortalamasını aldı
df.groupby("method")["orbital_period"].mean()

df.groupby("method")["orbital_period","mass"].mean()


ozet = df.groupby("method")["orbital_period","mass"].describe()



df.shape
df.head()
df.groupby("method")["method"].count()




df=pd.DataFrame(
    {"gruplar":["A","B","C","A","B","C"],
     "veri":[10,11,52,23,43,55]
     },
    columns = ["gruplar","veri"]
    )


df.groupby("gruplar").aggregate(["min",np.median,"max",np.mean])

df.groupby("gruplar").aggregate(["min",np.median,"max",np.mean])["veri"]




df.loc[(df.veri>10) & (df.gruplar=="A") | (df.gruplar=="B"),:]
df.loc[(df.veri>10) & (df.gruplar=="A") | (df.gruplar=="B"),:].groupby("gruplar").mean()


df1 = sns.load_dataset("planets")


df1.head()

df1.loc[:,["number","mass","orbital_period"]].dropna().apply(np.sum)

df1.loc[:,["number","mass","orbital_period"]].dropna().apply(np.mean)
df1.loc[:,["number","mass","orbital_period"]].dropna().apply(min)

df1.loc[:,["number","mass","orbital_period"]].groupby(df1.method).apply(np.sum)


# ders 144 ----------------------------------------------------------

import seaborn as sns

titanic = sns.load_dataset("titanic")

titanic.head()
titanic.shape
pd.Series(titanic.columns)


titanic.groupby(["sex","class"])[["survived"]].aggregate("mean").unstack()


#x eksenine sex sutunlara ise sınıfları koydu survived üzerine odaklandı.

titanic.pivot_table("survived",index="sex",columns="class")



age = pd.cut(titanic["age"],[0,18,90])
age.head()


titanic.corr()


#ders 234 ----------------------------------------------

import seaborn as sns
import numpy as np
import pandas as pd

df = sns.load_dataset("diamonds")
df.head()
df = df.select_dtypes(include = ["float64","int64"]) #sadece numericleri sec
df = df.dropna()
df.corr()
df.head()

diamonds = sns.load_dataset("diamonds")
df

#sadece string degerleri almak.
sns.load_dataset("diamonds").loc[:,(sns.load_dataset("diamonds").applymap(type)==str).all(0)]



df_table = df["table"]
df_table.head()

sns.boxplot(y=df_table)

q1 = df_table.quantile(0.25)
q3 = df_table.quantile(0.75)

IQR = q3-q1

alt_sinir = q1-1.5 * IQR
ust_sinir = q1+1.5 * IQR


df_table[(df_table<=ust_sinir) &(df_table>= alt_sinir)]

sns.boxplot(y = df_table[(df_table<=ust_sinir) &(df_table>= alt_sinir)])


index = np.random.randint(1,diamonds.shape[0],25)


diamonds.loc[index , ["depth"]] = np.nan


diamonds.loc[(diamonds.depth == np.nan), : ]

#eksik deger bulmak
diamonds.isnull().sum()

diamonds.head()

diamonds.loc[ (diamonds.depth == None), : ]

df_table


#Silme işlemi aykırı değer.
type(df_table)
df_table = pd.DataFrame(df_table)

df_table.shape

#tilda işareti olmayanları al demek
#axis 1 sutun bazında işlem yap.
t_df = df_table[~((df_table<alt_sinir) | (df_table>ust_sinir)).any(axis=1)]


# ortalama ile doldurma********************************************

df = sns.load_dataset("diamonds")
df.head()
df = df.select_dtypes(include = ["float64","int64"]) #sadece numericleri sec
df_table = df["table"]


sns.boxplot(y= df_table)

q1 = df_table.quantile(0.25)
q2 = df_table.quantile(0.75)

IQR = q2-q1

alt_sinir = q1-1.5 * IQR
ust_sinir = q1+1.5 * IQR




aykiri_index = df_table.loc[(df_table>ust_sinir) | (df_table<alt_sinir)].index


#bunlar benim aykırı değerlerim.
df_table[aykiri_index]
df_table[aykiri_index].describe()

#aykırı olmayan indeksler işime yaramaz zaten
set(aykiri_index) ^ set(df_table.index)

df_table[aykiri_index] = df_table.mean()

#aykırı degerleri ortalama ile doldurdum.

sns.boxplot(df_table)



#*************************************************************
#iki listenin kesişimi
 a = [1,2,3,4,5]
 b = [1,3,5,6]
 list(set(a) & set(b))
 #kesişşmeyen veriler
 set(a)^set(b)
#*************************************************************


#aykırı deger baskılama yontemi-----------
#ornegin 150 gibi bir degeriben ortalam ile değiştirirsem 
#150 yi 57 yapmış olurum bunun yerine yukarıda ise üst değerler
#ile aşagıda ise alt degerler ile değiştiririm


df = sns.load_dataset("diamonds")
df.head()
df = df.select_dtypes(include = ["float64","int64"]) #sadece numericleri sec
df.isnull().sum()
df_table = df["table"]


sns.boxplot(y= df_table)

q1 = df_table.quantile(0.25)
q2 = df_table.quantile(0.75)

IQR = q2-q1

alt_sinir = q1-1.5 * IQR
ust_sinir = q1+1.5 * IQR

#alt sınır değerinin altında kalan verileri
#alt sınır degeri ile baskıla
#üst sınır değerinin üstünde kalan verileri
#üst sınır degeri ile baskıla

df_table[df_table<alt_sinir] = alt_sinir

df_table[df_table>ust_sinir] = ust_sinir

sns.boxplot(y = df_table)


#Ders 236 cok degişkenli aykırı deger analizi.

veri = {
        "yaş" : pd.Series([17,18,70,80]),
        "evlilik":pd.Series([3,2,3,1])
        }


veri = pd.DataFrame(veri)

#bir kisi 17 yaşında 3 kere evlenemez çoklu olarak baktıgımda 
#bunu yakalayabiliyorum bu aykırı degerdir.
#genellikle karşılaşmayacagım bir durumdur.
#normalde 3 degeri aykırı deger degildir ama
#coklu olarak bakarsam aykırı deger

#bu yontem Local outlier factor yontemidir.
#bir noktanın local yogunlugu bu noktanın komşuları ile karşılastırılıyor.


df_table.hist()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20,contamination=0.1)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:20]
#sıralama fonksiyonu
np.sort(df_scores)






#Ders 238 Eksik veri ----------------------------------------------------

df.shape
df.head()
df.describe().T.loc[:, ["count","mean","std","min","max"]]

sns.boxplot(df_table)



#Ders 239 Eksik veri ----------------------------------------------------

import pandas as pd
import numpy as np

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df=pd.DataFrame(
    {"V1":V1,
     "V2":V2,
     "V3":V3
     }
    )

df.head()
df.isnull().sum()
df.notnull().sum() #boş olmayan
df.isnull().sum().sum() #toplam boş sayısı


df[1:5]
df[df.isnull().any(axis=1)] # en az 1 tane eksik veri olsa da al demek

df[df.notnull().all(axis=1)] # hepsi not null yani dolu olanları getir.


df.loc[(df.V1.notnull())  ,:]
df.loc[(df.V1.notnull()) & (df.V2.notnull())  ,:]
df.loc[(df.V1.notnull()) & (df.V2.notnull()) & (df.V3.notnull())  ,:]

df[df.V1.isnull()]


df

df.isnull().sum() / df.shape[0] * 100 # eksik veri yüzdelik.

df.loc[df.V1.isnull() , : ]

df.dropna() #sadece tam olanları getir.


df.loc[df.V1.isnull() , "V1"]

df.loc[df.V2.isnull() , "V2"]

df["V1"].fillna(df["V1"].mean()) #ortalama ile doldurdu.

df["V2"].fillna(0)

#surekli boyle tek tek ortalama ile doldurmam zaman alır
#apply fonksiyonu ile sutunlar içerisinde gezecegim

# x her bir sutunu ifade eder axis 0 ise sutunu ifade eder sutun ortalamasını al
df.apply(lambda x : x.fillna(x.mean()) , axis=0)

df.apply(lambda x : x.isnull().sum() / len(x) ,axis=0)
df.apply(lambda x : x.isnull().sum() / len(x) ,axis=1)


df.apply(lambda x : x.count()) #her sutunun dolu degerini sayar.


#Ders 240 Eksik veri görselleştirme************************************

df.isnull().sum()
df.isnull().sum().sum()

df.dropna().apply(lambda x :x+10 , axis=0)
df.apply(lambda x : x.isnull().sum() / len(x))
df.V1.fillna(df.V1.mean())

for i in df.columns:
    print(type(df[i]))


import missingno as msno
msno.bar(df); #eksik degerleri oransal olarak ifade ediyor.
#sol taraf yüzdelşik dilimi sağ taraf adedi verir.

msno.matrix(df)
#sksik verininrassallıgını gosterir missmap fonksiyonu gibi
df

import seaborn as sns

df=sns.load_dataset("planets")

df.shape
df.head()
df.count()
df.isnull().sum()


msno.matrix(df)
#orbital değişkeninde ne zaman bir eksiklik olsa  mass de de var.
#orbital period değişkeni mass ile bağımlı.


msno.heatmap(df);
#ısı haritası 
#bu harita bize nullitycorelastion degerini verir.
#yani eğer iki değişkenin korelasyon degeri 1 ise
#birinde eksiklik olunca digerşnde de var demektşr.
# 0 ise birbirlerini etkileyen korelasyon yoktur demektir.
#bu veri seti rassal bir eksikliğe sahip değildir.
#direk doldurulamaz.
msno.matrix(df)
msno.heatmap(df)

df.head()


#Ders 241 ***********************************************************

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df=pd.DataFrame(
    {"V1":V1,
     "V2":V2,
     "V3":V3
     }
    )


df

a=np.arange(0,df.shape[0])
a=pd.Series(a)
df = pd.concat([df,a],axis=1)


df.dropna()
#hepsi aynı anda eksik olan verileri sil.
df.dropna(how = "all")


# eger tum degerleri dolu olan bir değiken olsaydı kalırdı
df.dropna(axis=1)

df=df.drop(0,axis=1)
df

#axis 0 satırı 1 sutunu temsil eder.

#yeni kolon ekledim
df["sil_beni"] = np.nan



df

#degiskeninin tüm degerleri nan olan degerleri sil
df = df.dropna(how = "all",axis = 1)


df.isnull().sum()
msno.heatmap(df)
msno.matrix(df)

msno.matrix(df.dropna(how= "all"))


df

#Ders 242 sayısal degişkenlerde atama işlemi
df
df.dropna()
df.loc[:, "V1"].fillna(0)
df.loc[:, "V1"].fillna(df.V1.mean())
df.apply(lambda x : x.fillna(x.mean())) #tum degiskenler için birinci yol

df.dropna(how="all")

df.fillna(df.mean())  #tum degiskenler için ikinci yol

#verinin dağılımına gore doldur normal dağılmıisa ortalama normal dağılmamışsa 
#medyan ile doldur.

df.loc[:, ["V1","V2"]].fillna(df.mean()) #1 ve 2 yi ortalama ile doldurdu.

df["V3"].fillna(df.V3.median())



df.isnull().sum()

df["V1"]

df
df.loc[df.V1.isnull() , :]
df["V1"].fillna(df.V1.mean())


df


planet = sns.load_dataset("planets")

planet.groupby("method")["method"].count()

len(planet.loc[planet.method == "Radial Velocity",:])
planet.columns
planet.describe().T[["count","mean","min","max"]]
planet.groupby("method")["number"].aggregate(["count","min","mean"])


planet.isnull().sum()

planet.loc[(planet.orbital_period.isnull()) & (planet.distance.isnull()),:]

planet.isnull().sum()



import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib as mp
df=sns.load_dataset("planets")


df.head()
df.shape
df.describe().T[["min","max","mean","std"]]
df.count()
df.isnull().sum()
df.columns

#Dummy variables

df_categorycal = pd.get_dummies(df)
df_categorycal.shape
df.groupby("method")["method"].count()

pd.Series(df_categorycal.columns)
pd.Series(pd.unique(df.method))
len(pd.Series(pd.unique(df.method)))

df_categorycal.isnull().sum().sum()

df_categorycal.apply(lambda x : x.isnull().sum() / len(x))

df.loc[ (df.mass.isnull()) , :]

msno.matrix(df_categorycal)
msno.heatmap(df_categorycal)

sns.histplot(df.mass)
sns.boxplot(y = df.mass)

df.mass.quantile(0)
df.mass.quantile(1)
df.mass.quantile(0.75)
df.mass.describe()


alt_sinir = df.mass.quantile(0.25)
ust_sinir = df.mass.quantile(0.75)
fark = df.mass.quantile(0.75) - df.mass.quantile(0.25)

alt_sinir - 1.5 * fark
ust_sinir + 1.5 * fark


df.loc[ (df.mass>alt_sinir - 1.5 * fark) & (df.mass<ust_sinir + 1.5 * fark) , "mass"]

sns.boxplot(y=df.mass)
sns.boxplot(y=df.loc[(df.mass>=alt_sinir - 1.5 * fark)
                     & (df.mass<=ust_sinir + 1.5 * fark) , "mass"]
)

sns.histplot(df.mass)

sns.histplot(df.loc[(df.mass>=alt_sinir - 1.5 * fark) & 
                                  (df.mass<=ust_sinir + 1.5 * fark) , "mass"]
)


(df.loc[(df.mass>=alt_sinir - 1.5 * fark) & 
                                  (df.mass<=ust_sinir + 1.5 * fark) , "mass"]
).mean()
df.mass.mean()



df.mass.fillna(df.mass.median())

#eksik veri doldururken ortalama maaşı atamak yerine ar-ge de çalışanlara
# ar-ge departmanonın maaşını atarken depo bölümünde çalışanlara depo
#bölümünün ortalama maaşoını atamak çok daha mantıklı hatta departman içerisindekilere
#de tecrübelerine göre ata

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3=np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4=np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])


df=pd.DataFrame({
    "Maas":V1,
    "V2":V2,
    "V3":V3,
    "departman":V4
    })


df.head()
df.count()/df.shape[0]

df.groupby("departman")["Maas"].mean()

df["Maas"] = df.Maas.fillna(df.groupby("departman")["Maas"].transform("mean"))

#burada her departmanın ortalamsını boş olan degerlere atadı.

#eger ben genel ortalamayı atarsam olmaz.
df


#ders 244 kategorik değişkende atama yaomak.


V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.NaN,"IK","IK","IK","IK","IK","IT","IT"],dtype=object)

df=pd.DataFrame({
    "Maas":V1,
    "departman":V4
    })
df

df["departman"].mode()[0]
df.departman.fillna(df["departman"].mode()[0],inplace=True)

#inplace true dedim kalıcı olarak kaydetti


#Ders 245 Tahmine dayalı değer atama.*********************************

df=sns.load_dataset("titanic")
df=df.select_dtypes(include = ["float64","int64"])
df.head()
df.shape
df.isnull().sum()


from ycimpute.imputer import knnimput

var_names = list(df) #değişken isimlerini aldı.

#knn ile atama yapabilmek için burada matrix kullanmalıyım
#bu yüzdem veri setinin numpy arrayine dönüştürüyorum

n_df = np.array(df)

n_df[0:5]
n_df.shape

dff = knnimput.KNN(k=4).complete(n_df)

type(dff)


dff=pd.DataFrame(
    dff,
    columns=var_names
    )

dff.head()

dff.isnull().sum()
df.isnull().sum()

#randomforest ile doldurma. ******************************************

from ycimpute.imputer import iterforest
df=sns.load_dataset("titanic")
df=df.select_dtypes(include = ["float64","int64"])

df.isnull().sum()

var_names = list(df) 

n_df = np.array(df)

dff = iterforest.IterImput().complete(n_df)

dff= pd.DataFrame(dff,columns=var_names)

dff.head()


#Veri standartizasyonu ********************************************

dff.head()

#ortalama standart sapma dönüşümü yaptım.
dff.apply(lambda x : (x-x.mean())  / x.std() )

from sklearn import preprocessing #aynı işlemi yapar.
preprocessing.scale(dff)



#min max donusumu yaptım
dff.apply(lambda x : (x-min(x))/(max(x)-min(x)) ).head()



#normalizasyondur 0-1 arası dönüştürür ama ben kendi istediğim 
#min mak değere de dönüştürebilirim.
preprocessing.normalize(dff)[0:5]


#ben 0-1 aralığında değilde ben 

scaler = preprocessing.MinMaxScaler(feature_range = (10,20))
scaler.fit_transform(df)

#değişkenleri 10 - 20 arasında boldu.


#Ders 247 değişken dönüşümleri ********************************

df=sns.load_dataset("tips")

df.head()







