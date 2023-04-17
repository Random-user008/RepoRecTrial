import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.components.v1 import components

# Define vectorizer with desired parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto')


def extract_topic_distributions(texts, num_topics=10):
    # Convert text data to document-term matrix
    dtm = vectorizer.fit_transform(texts)

    # Extract topics from document-term matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    topic_distributions = lda.transform(dtm)

    return topic_distributions


def get_top_topics(descriptions, num_topics=10):
    # Convert text data to document-term matrix
    dtm = vectorizer.fit_transform(descriptions)

    # Extract topics from document-term matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)

    # Get the top topics
    top_topics = []
    for i in range(num_topics):
        top_topic_words = [vectorizer.get_feature_names()[index] for index in lda.components_[i].argsort()[-5:]]
        top_topics.append(', '.join(top_topic_words))

    return top_topics


def get_top_starred_repos():
    top_repos_data = pd.read_csv('https://raw.githubusercontent.com/Random-user008/FinFiles/master/top_starred_repos.csv')
    return top_repos_data


def get_top_n_indices(topic_distributions, top_n=5):
    # Get top n indices with highest topic probability sum
    topic_probs = topic_distributions.sum(axis=1)
    top_n_indices = topic_probs.argsort()[-top_n:][::-1]

    return top_n_indices  


def elbow_method_plot(model, data):
    # Fit the KMeans model to the data
    model.fit(data)

    # Calculate the within-cluster sum of squares (inertia) for each value of k
    k_range = range(1, 11)
    inertias = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    # Plot the elbow curve
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method Plot')
    plt.show()

def recommend_repos(username):

    user_repos = requests.get(f'https://api.github.com/users/{username}/repos',).json()
    user_repo_descriptions = [repo['description'] for repo in user_repos if repo['description']]
    user_repo_languages = [repo['language'] for repo in user_repos if repo['language']]
    user_repo_topics = [",".join(repo['topics']) for repo in user_repos if repo['topics']]
    print()
    top_repos_data = get_top_starred_repos()
    top_repo_descriptions = top_repos_data['description'].tolist()
    top_repo_languages = top_repos_data['language'].tolist()
    top_repo_topics = top_repos_data['topics'].tolist()
    # Remove np.nan values from descriptions
    top_repo_descriptions = [desc for desc in top_repo_descriptions if isinstance(desc, str)]
    top_repo_languages = [lang for lang in top_repo_languages if isinstance(lang, str)]
    top_repo_topics = [topic for topic in top_repo_topics if isinstance(topic,str)]
    # Concatenate user repository descriptions with top repository descriptions
    texts = user_repo_descriptions + top_repo_descriptions
    languages = user_repo_languages + top_repo_languages
    topics  = user_repo_topics +  top_repo_topics
    # Check if there are any repository descriptions available
    if len(texts) == 0:
        print("No repository descriptions available for recommendation")
        return
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(texts)
    
    # Extract topic distributions and language encodings for the text data
    topic_distributions = extract_topic_distributions(texts)
    language_encodings = vectorizer.transform(languages)
    topics_encoding = vectorizer.transform(topics)
    n = max(topic_distributions.shape[0], language_encodings.shape[0],topics_encoding.shape[0])
    topic_distributions_pad = np.pad(topic_distributions, ((0, n - topic_distributions.shape[0]), (0, 0)))
    language_encodings_pad = np.pad(language_encodings.toarray(), ((0, n - language_encodings.shape[0]), (0, 0)))
    topics_pad = np.pad(topics_encoding.toarray(),((0, n - topics_encoding.shape[0]), (0, 0)))
    data = np.concatenate((topic_distributions_pad, language_encodings_pad,topics_pad), axis=1)

    # Combine topic distributions and language encodings
    # data = np.concatenate((topic_distributions, language_encodings.toarray()), axis=1)
    
    # Use the elbow method to find the optimal number of clusters for K-Means
    kmeans = KMeans(random_state=42)
    elbow_method_plot(kmeans, data)
    
    # Cluster the data using K-Means
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)
    
    # Find the top N repositories in each cluster
    top_n = 5
    top_n_indices = []
    
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_scores = [kmeans.score([data[j]]) for j in cluster_indices]
        top_n_indices.extend([cluster_indices[j] for j in np.argsort(cluster_scores)[-top_n:]])
    
    # Filter the top N repositories from the top starred repositories
    if len(top_repos_data) < top_n:
        top_n_repos = top_repos_data
    else:
        top_n_repos = top_repos_data.iloc[top_n_indices]
    
    st.write(f"Top {len(top_n_repos)} recommended repos:")
    st.write(top_n_repos[['name', 'description', 'language', 'url','topics']])

with st.form(key = "form1"):
     Username = st.text_input('Enter GitHub Username')
     submit = st.form_submit_button(label = "Submit")
     if submit:
        recommend_repos(Username)
