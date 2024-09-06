from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from transformers import CLIPProcessor, CLIPModel

def main():
    # Clustering algorithm
    # Prepare data
    corpus = ["hãy sống hết mình mỗi ngày như thể đó là ngày cuối cùng của hành trình cuộc đời và hãy học hỏi như thể bạn sẽ sống mãi mãi",
            "hãy tin tưởng vào bản thân và vào hành trình cuộc sống mọi điều sẽ trở nên tốt đẹp hơn khi bạn có lòng tin",
            "cuộc sống không chờ đợi mưa ngừng mà là việc học cách sống vui vẻ dưới cơn mưa",
            "học toán online hiệu quả với hàng ngàn video bài giảng, bài tập luyện tập từ cơ bản đến nâng cao",
            "trong chuyên mục 5 phút học toán hôm nay, thầy Hiếu giới thiệu bài toán rất hay và khó về tính tổng các chữ số",
            "với tâm trí quyết tâm và không bỏ cuộc rằng ước mơ sẽ hóa thành sự thật",

            "unlike Icelandic and Faroese which were isolated the development of English was influenced by a long series of invasions",
            "these left a profound mark of their own on the language so that English shows some similarities in vocabulary",
            "typical wiki contains multiple pages that can either be edited by the public or limited to use within an organization for",
            "have little inherent structure, allowing one to emerge according to the needs of the users",
            "hosting user-authored content wikis allow those users to interact, hold discussions and collaborate",
            "English is classified as a Germanic language because it shares innovations with other Germanic languages"]

    # Scale data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.vocabulary_)
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names)

    print(type(X))
    X = X.toarray()
    print(len(X))

    # pair plot
    # ???
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(X_pca)
    df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    sns.pairplot(df)
    plt.show()

    # word cloud
    text_combined = ' '.join(corpus)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    print(wordcloud)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Model instantiation
    # k=2
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    print(labels)
    print(kmeans.inertia_)

    for sentence, label in zip(corpus, labels):
        print(f"Cluster {label}: {sentence[:50]}...")

    # optimal k using elbow on wcss score
    wcss_values = []
    for i in range(1, 9):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        wcss = kmeans.inertia_
        wcss_values.append(wcss)

    # Khi plot len gia wcss thay doi chua ro rang trong truong hop nay co the do du lieu khong du lon
        
    plt.plot(range(1, 9), wcss_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters')
    plt.show()

    # Applying magic function
    def magic_function(corpus):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(text=corpus, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        text_features = text_features.numpy()
        print(text_features.shape)

        return text_features
    
    # get new features
    text_features = magic_function(corpus)
    print(text_features)
    print(text_features[0])

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(text_features)

    # print 
    for sentence, label in zip(corpus, kmeans.labels_):
        print(f"Cluster {label}: {sentence[:50]}...")

    #  WCSS 
    wcss_values = []
    for i in range(1, 9):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(text_features)
        wcss_values.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 9), wcss_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters (CLIP embeddings)')

    plt.show()

if __name__ == "__main__":
    main()
