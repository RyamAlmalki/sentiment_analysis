import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Task 1
def sentimentAnalyzer(text):
    blob = TextBlob(str(text))  
    polarity = blob.sentiment.polarity
    if polarity < -0.2:
        return 'Negative'
    elif -0.2 <= polarity <= 0.2:
        return 'Neutral'
    else:
        return 'Positive'

# Task 2
def verifyFunction():
    words = ['happy', 'exciting', 'good', 'rich', 'smile', 'sad', 'disappointed', 'bad', 'poor', 'anger', 'food', 'animal']

    for word in words:
        print("Word is {} it's result {}".format(word, sentimentAnalyzer(word)))



# Task 3
data = pd.read_csv('Amazon_Unlocked_Mobile.csv')

product_filter = "Apple iPhone 4s 8GB Unlocked Smartphone w/ 8MP Camera, White (Certified Refurbished)"
specific_product_data = data[data['Product Name'] == product_filter]

# Task 4
specific_product_data['Sentiment'] = specific_product_data['Reviews'].apply(sentimentAnalyzer)



# Task 5
plt.figure(figsize=(8, 6))
specific_product_data['Sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis of Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.show()


print("\nInsights for Minimizing Negative Sentiment:")
negative_reviews = specific_product_data[specific_product_data['Sentiment'] == 'Negative']

common_complaints = negative_reviews['Reviews'].str.extractall(r'(slow|battery|crash|buggy|freeze)')[0].value_counts()
print(common_complaints)

print("\nPotential Misclassifications:")
misclassified_reviews = specific_product_data[
    (specific_product_data['Sentiment'] == 'Positive') & (specific_product_data['Rating'] < 3)
]
print(misclassified_reviews[['Reviews', 'Rating', 'Sentiment']])

# Task 6
output_file_path = 'Apple_iPhone_4s_White_With_Sentiments.csv'
specific_product_data.to_csv(output_file_path, index=False)

print(f"\nThe file with sentiments for the selected product has been saved to {output_file_path}")
