#DATA LOADING
import pandas as pd
# pd.set_option('display.max_columns', None) #code to prevent pandas from truncating columns.
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('Spotify_data.csv', na_values=['NaN', 'NULL', 'null', 'NA', 'n/a', ''], keep_default_na=False)

print(df.head())
print()

#DATA CLEANING
print(df.isna().sum())
print()

#DATA SUMMARIZATION.
summary = df.groupby(['spotify_subscription_plan', 'fav_music_genre']).size()
print(summary.sort_values(ascending=False))
print()


#EDA
desc = df.describe(include='all')
print(desc.loc[['count', 'unique', 'top', 'freq']]) #This limits the decription to a few rows
print()

#1. what age group dominates the platform?
print(df['Age'].value_counts())
print()

#2. what spotify listening plan is the most common?
print(df['spotify_subscription_plan'].value_counts())
print()

#3. For premium user, what kind of premium plan do they prefer?
print(df['preffered_premium_plan'].value_counts())
print()

#4. what are the most popular music genre being lsitened to?
print(df['fav_music_genre'].value_counts())
print()

#VISUALIZATION

#Subscription plan
df['spotify_subscription_plan'].value_counts().plot(kind='bar')
plt.title('Subscription Plan Distribution')
plt.xlabel('Plan')
plt.ylabel('Number of Users')
plt.xticks(rotation=10, ha='right')
plt.savefig("spotify_subscription_plot.png")

#Music Genre
df['fav_music_genre'].value_counts().plot(kind='bar')
plt.title('Music Genre Distribution')
plt.xlabel('Plan')
plt.ylabel('Number of Users')
plt.xticks(rotation=12, ha='right')
plt.savefig("fav_music_genre.png")

#MODEL AND TESTING
# Features
X = df.drop('spotify_subscription_plan', axis=1)

# Target
y = df['spotify_subscription_plan']

# Convert text columns to numbers
X = pd.get_dummies(X)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))








