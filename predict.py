#!/usr/bin/env python3
import joblib as jb
import pandas as pn

model = jb.load("model/xgb_model.pkl")
FEATURE_ORDER = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus_furnished", "furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"]

def classify_property(input_features, actual_price, rmse=1617760, k=0.7):
	input_features = input_features[FEATURE_ORDER]
	prediction = model.predict(input_features)[0]
	threshold = k * rmse
	gap = actual_price - prediction
	if prediction != 0:
		percentage_gap = (gap / prediction) * 100
	else:
		percentage_gap = 0

	if abs(gap) <= threshold:
		label = "Fair"
	else:
		if gap < 0:
			if abs(percentage_gap) < 20:
				label = "Slightly Underpriced"
			else:
				label = "Strongly Underpriced"
		else:
			if abs(percentage_gap) < 20:
				label = "Slightly Overpriced"
			else:
				label = "Strongly Overpriced"

	return prediction, gap, percentage_gap, label


if __name__ == "__main__":
	features = pn.DataFrame([{"area": 6500, "bedrooms": 3, "bathrooms": 2, "stories": 3, "mainroad": 1, "guestroom": 0, "basement": 0, "hotwaterheating": 0, "airconditioning": 1, "parking": 0, "prefarea": 1, "furnishingstatus_furnished": 1, "furnishingstatus_semi-furnished": 0, "furnishingstatus_unfurnished": 0 }])
	actual_price = 6650000

	prediction, gap, gap_percentage, label = classify_property(features, actual_price)
	print("Actual price: ", actual_price)
	print("Prediction: ", prediction)
	print("Gap: ", gap)
	print("Gap percentage: ", gap_percentage)
	print("label: ", label)
