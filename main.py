import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

#verilerin yüklenmesi
veriler = pd.read_csv('ETicaret.csv')

ulkeler = veriler.iloc[:,1]

X = veriler.iloc[:,2:5]
Y = veriler.iloc[:,[5]]

#kategorik verilerin encode edilmesi
le = LabelEncoder()
ulkeler= le.fit_transform(ulkeler)
print(ulkeler)

ohe = OneHotEncoder()
ulkeler = ohe.fit_transform(ulkeler.reshape(-1,1)).toarray()
print(ulkeler)

#encode işlemi sonucu oluşan sutünların dataframe dönüştürülmesi
sonuc = pd.DataFrame(data=ulkeler, index= range(101), columns=['ABD','fransa','ingiltere','turkiye'] )
print(sonuc)
sonuc2 = pd.concat([sonuc, X], axis=1)
print(sonuc2)

#eğitim ve test verilerin ayrıştırılması
x_train, x_test, y_train, y_test = train_test_split(sonuc2, Y, test_size=0.33, random_state=0)

#verilerin normalize edilmesi
min_max_scale = MinMaxScaler()
x_train = min_max_scale.fit_transform(x_train)
x_test = min_max_scale.transform(x_test)

#model eğitim
lr = LinearRegression()
lr.fit(x_train, y_train)

#tahmin ve başarı
y_pred = lr.predict(x_test)

print(y_pred)
print(r2_score(y_test, y_pred))