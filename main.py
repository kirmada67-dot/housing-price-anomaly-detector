#!/usr/bin/env python3
import pandas as pn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pn.read_csv("Housing.csv")
df["mainroad"] = df["mainroad"].map({"yes": 1, "no": 0})
df["guestroom"] = df["guestroom"].map({"yes": 1, "no": 0})
df["basement"] = df["basement"].map({"yes": 1, "no": 0})
df["hotwaterheating"] = df["hotwaterheating"].map({"yes": 1, "no": 0})
df["airconditioning"] = df["airconditioning"].map({"yes": 1, "no": 0})
df["prefarea"] = df["prefarea"].map({"yes": 1, "no": 0})
df = pn.get_dummies(df, columns=["furnishingstatus"])
df["furnishingstatus_furnished"] = df["furnishingstatus_furnished"].map({True: 1, False: 0})
df["furnishingstatus_semi-furnished"] = df["furnishingstatus_semi-furnished"].map({True: 1, False: 0})
df["furnishingstatus_unfurnished"] = df["furnishingstatus_unfurnished"].map({True: 1, False: 0})

x = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus_furnished", "furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Mdoel: ", model)
print("MSE: ", mean_squared_error(y_test, pred))
print("r2_score: ", r2_score(y_test, pred))
