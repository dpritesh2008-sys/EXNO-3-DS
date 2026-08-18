## EXNO-3-DS
## Name : Ritesh DP
## Reg.No. : 212225040339

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

~~~

# ============================================================
# EX NO 3 - KVP
# ENCODING AND FEATURE TRANSFORMATION
# ============================================================

# -----------------------------
# ENCODING
# -----------------------------

import pandas as pd

# Read Encoding Data
df = pd.read_csv("/content/Encoding Data.csv")
df


# -----------------------------
# ORDINAL ENCODING
# -----------------------------

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm = ['Hot', 'Warm', 'Cold']

e1 = OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])


df['bo2'] = e1.fit_transform(df[["ord_2"]])
df


# -----------------------------
# LABEL ENCODING
# -----------------------------

le = LabelEncoder()

dfc = df.copy()

dfc['ord_2'] = le.fit_transform(dfc['ord_2'])

dfc


# -----------------------------
# ONE HOT ENCODING
# -----------------------------

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

df2 = df.copy()

enc = pd.DataFrame(
    ohe.fit_transform(df2[["nom_0"]])
)

df2 = pd.concat([df2, enc], axis=1)

df2


# Using get_dummies
pd.get_dummies(df2, columns=["nom_0"])


# -----------------------------
# INSTALL CATEGORY ENCODERS
# -----------------------------

!pip install --upgrade category_encoders


# -----------------------------
# BINARY ENCODER
# -----------------------------

from category_encoders import BinaryEncoder

df = pd.read_csv("/content/data.csv")
df


be = BinaryEncoder()

nd = be.fit_transform(df['Ord_2'])

dfb = pd.concat([df, nd], axis=1)

dfb


# -----------------------------
# MEAN ENCODING
# -----------------------------

from category_encoders import TargetEncoder

te = TargetEncoder()

CC = df.copy()

new = te.fit_transform(
    X=CC["City"],
    y=CC["Target"]
)

CC = pd.concat([CC, new], axis=1)

CC


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================

import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("/content/Data_to_Transform.csv")
df


# Check skewness
df.skew()


# -----------------------------
# 1. LOG TRANSFORMATION
# -----------------------------

np.log(df["Highly Positive Skew"])


# -----------------------------
# 2. RECIPROCAL TRANSFORMATION
# -----------------------------

np.reciprocal(df["Moderate Positive Skew"])


# -----------------------------
# 4. SQUARE ROOT TRANSFORMATION
# -----------------------------

np.sqrt(df["Highly Positive Skew"])


# -----------------------------
# 5. SQUARE TRANSFORMATION
# -----------------------------

np.square(df["Highly Positive Skew"])


# -----------------------------
# POWER TRANSFORMATIONS
# -----------------------------

# BOX COX

df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(
    df["Highly Positive Skew"]
)

df


# Check skewness after Box-Cox
df.skew()


# -----------------------------
# YEO-JOHNSON
# -----------------------------

df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(
    df["Highly Negative Skew"]
)

df.skew()


# -----------------------------
# QUANTILE TRANSFORMATION
# -----------------------------

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"] = qt.fit_transform(
    df[["Moderate Negative Skew"]]
)

df


# ============================================================
# QQ PLOT
# ============================================================

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt


# QQ plot - original data
sm.qqplot(
    df["Moderate Negative Skew"],
    line='45'
)

plt.show()


# QQ plot - reciprocal transformation
sm.qqplot(
    np.reciprocal(df["Moderate Negative Skew"]),
    line='45'
)

plt.show()


# -----------------------------
# QUANTILE TRANSFORMATION
# FOR QQ PLOT
# -----------------------------

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(
    output_distribution='normal',
    n_quantiles=891
)

df["Moderate Negative Skew"] = qt.fit_transform(
    df[["Moderate Negative Skew"]]
)


# Final QQ plot
sm.qqplot(
    df["Moderate Negative Skew"],
    line='45'
)

plt.show()
~~~

<img width="662" height="724" alt="Screenshot 2026-08-06 083555" src="https://github.com/user-attachments/assets/f4ac3c4b-c84b-46dc-b5b7-6d48c9aa7887" />
<img width="427" height="768" alt="Screenshot 2026-08-06 083609" src="https://github.com/user-attachments/assets/c9041d56-6e15-4bfa-b01f-0847755e9983" />
<img width="476" height="722" alt="Screenshot 2026-08-06 083621" src="https://github.com/user-attachments/assets/9c0cfd1a-ccdb-4b69-acfb-40b9b3d66092" />
<img width="631" height="782" alt="Screenshot 2026-08-06 083631" src="https://github.com/user-attachments/assets/8a1d6724-dcbb-4bc4-aa4d-9a6afed3a18a" />
<img width="1151" height="780" alt="Screenshot 2026-08-06 083644" src="https://github.com/user-attachments/assets/2c57fa06-1408-4995-abc4-b2cc3291ed7d" />
<img width="697" height="754" alt="Screenshot 2026-08-06 083655" src="https://github.com/user-attachments/assets/5d667f38-3f24-450f-93da-5a87d721118f" />
<img width="714" height="769" alt="Screenshot 2026-08-06 083705" src="https://github.com/user-attachments/assets/90dfc4a9-3def-4dd5-8ef5-c51183705c81" />
<img width="885" height="780" alt="Screenshot 2026-08-06 083715" src="https://github.com/user-attachments/assets/76ac2789-5dce-4d92-b5c3-022140a4e245" />
<img width="549" height="770" alt="Screenshot 2026-08-06 083725" src="https://github.com/user-attachments/assets/0a04d462-8062-4d96-b2e7-85aa25155f14" />
<img width="547" height="761" alt="Screenshot 2026-08-06 083736" src="https://github.com/user-attachments/assets/7ddc4ee4-b3de-4414-8ffb-ca620a5533d1" />
<img width="1162" height="784" alt="Screenshot 2026-08-06 083751" src="https://github.com/user-attachments/assets/21965c7a-f683-4094-8b69-5e4c7c242162" />
<img width="1137" height="797" alt="Screenshot 2026-08-06 083800" src="https://github.com/user-attachments/assets/05e5e390-b612-4ad9-91b9-e019d76ebe28" />
<img width="1490" height="781" alt="Screenshot 2026-08-06 083812" src="https://github.com/user-attachments/assets/9ba2bf1f-8cdb-449a-acf1-d883cb60db3c" />
<img width="1486" height="723" alt="Screenshot 2026-08-06 083829" src="https://github.com/user-attachments/assets/465cf38c-2a4e-4f79-b24d-6125a54b70b4" />
<img width="621" height="492" alt="Screenshot 2026-08-06 083840" src="https://github.com/user-attachments/assets/13603e75-3eee-4891-ba0b-6cf993e41dd1" />
<img width="734" height="603" alt="Screenshot 2026-08-06 083847" src="https://github.com/user-attachments/assets/92cc437a-71d2-4c92-9180-f1f55f2c8561" />
<img width="649" height="609" alt="Screenshot 2026-08-06 083859" src="https://github.com/user-attachments/assets/76ec47eb-a204-4366-9eda-3c3589d5492a" />
<img width="742" height="495" alt="Screenshot 2026-08-06 083909" src="https://github.com/user-attachments/assets/ca96af63-249c-4fdb-8eac-1b7f6558f9d5" />
<img width="658" height="675" alt="Screenshot 2026-08-06 083920" src="https://github.com/user-attachments/assets/5dfcc2d4-0769-41e3-ae7b-2197bd283bce" />
<img width="713" height="486" alt="Screenshot 2026-08-06 084017" src="https://github.com/user-attachments/assets/4fb21093-f511-416c-96bc-c9f701f67f52" />
<img width="1218" height="746" alt="Screenshot 2026-08-06 084030" src="https://github.com/user-attachments/assets/a2480606-345b-48ac-ab0a-161097a4f7ba" />
       
# RESULT:
     
~~~
Successfully read the given data, performed feature encoding and transformation, and saved the transformed data to a file.
~~~
       
