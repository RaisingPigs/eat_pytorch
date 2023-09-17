import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


x_value = np.random.randint(140, 180, 200)

plt.figure(figsize=(20, 8), dpi=80)
plt.hist(x_value, bins=10, density=1)

plt.title("data analyze")
plt.xlabel("height")
plt.ylabel("rate")
plt.show()
