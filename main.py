#!/usr/bin/env python3
import pandas as pn
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
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

model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42)
model.fit(x_train, y_train)
pipeline = Pipeline([("model", model)])
scores = cross_val_score(pipeline, x, y, cv=5, scoring='neg_mean_squared_error')
mse = -scores.mean()
rmse = np.sqrt(mse)
print("Model: ", model)
print("RMSE: ", rmse)

features = pn.DataFrame([{"area": 6500, "bedrooms": 3, "bathrooms": 2, "stories": 3, "mainroad": 1, "guestroom": 0, "basement": 0, "hotwaterheating": 0, "airconditioning": 1, "parking": 0, "prefarea": 1, "furnishingstatus_furnished": 1, "furnishingstatus_semi-furnished": 0, "furnishingstatus_unfurnished": 0 }])
actual_price = 6650000

def classify_property(model, input_features, actual_price, rmse, k=0.7):
	prediction = model.predict(input_features)[0]
	threshold = k * rmse
	gap = actual_price - prediction
	if prediction != 0:
		percentage_gap = (gap / prediction) * 100
	else:
		percentage_gap = 0

	if abs(gap) <= threshold:
		label = "fair"
	else:
		if gap < 0:
			if abs(percentage_gap) < 20:
				label = "slightly underpriced"
			else:
				label = "strongly underpriced"
		else:
			if abs(percentage_gap) < 20:
				label = "slightly overpriced"
			else:
				label = "strongly overpriced"

	return prediction, gap, percentage_gap, label

prediction, gap, gap_percentage, label = classify_property(model, features, actual_price, rmse, k=0.7)
print("Actual price: ", actual_price)
print("Prediction: ", prediction)
print("Gap: ", gap)
print("Gap precentage: ", gap_percentage)
print("label: ", label)
