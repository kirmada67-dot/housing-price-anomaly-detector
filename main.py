#!/usr/bin/env python3
import pandas as pn
import matplotlib.pyplot as plt
df = pn.read_csv("Housing.csv")
plt.hist(df["area"], bins=30)
plt.show()
plt.hist(df["price"], bins=30)
plt.show()
