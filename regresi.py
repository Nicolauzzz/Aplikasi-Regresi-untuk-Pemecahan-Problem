import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Data
TB = np.array([7, 4, 8, 5, 7, 3, 7, 8, 5]).reshape(-1, 1)  # Durasi waktu belajar
NT = np.array([91, 65, 45, 36, 66, 61, 63, 42, 61])  # Nilai ujian

# Model linear (Metode 1)
model_linear = LinearRegression()
model_linear.fit(TB, NT)
NT_pred_linear = model_linear.predict(TB)

# Model pangkat sederhana (Metode 2)
poly_features = PolynomialFeatures(degree=2)
TB_poly = poly_features.fit_transform(TB)
model_poly = LinearRegression()
model_poly.fit(TB_poly, NT)
NT_pred_poly = model_poly.predict(TB_poly)

# Plotting
plt.figure(figsize=(12, 6))

# Plot data dan regresi linear
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, color='blue', label='Data')
plt.plot(TB, NT_pred_linear, color='red', label='Regresi Linear')
plt.title('Regresi Linear')
plt.xlabel('Durasi Waktu Belajar')
plt.ylabel('Nilai Ujian')
plt.legend()

# Plot data dan regresi pangkat sederhana
plt.subplot(1, 2, 2)
plt.scatter(TB, NT, color='blue', label='Data')
plt.plot(TB, NT_pred_poly, color='red', label='Regresi Pangkat Sederhana')
plt.title('Regresi Pangkat Sederhana (Deg=2)')
plt.xlabel('Durasi Waktu Belajar')
plt.ylabel('Nilai Ujian')
plt.legend()

plt.tight_layout()
plt.show()

# Hitung galat RMS
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_poly = np.sqrt(mean_squared_error(NT, NT_pred_poly))

print("Galat RMS Regresi Linear:", rms_linear)
print("Galat RMS Regresi Pangkat Sederhana:", rms_poly)
