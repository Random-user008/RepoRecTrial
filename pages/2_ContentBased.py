import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import NMF


components.html("""
        <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');</style>
        <h2 style="color:White;font-family: 'Roboto Mono', monospace;
    font-size:20px">Enter the details to Recommend Projects </h2>    """,height=60,width=700)

# Define vectorizer with desired parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto')


# def extract_topic_distributions(texts, num_topics=10):
#     # Convert text data to document-term matrix
#     dtm = vectorizer.fit_transform(texts)

#     # Extract topics from document-term matrix
#     lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
#     lda.fit(dtm)
#     topic_distributions = lda.transform(dtm)

#     return topic_distributions

def extract_topic_distributions(texts, num_topics=10):
    # Convert text data to document-term matrix
    dtm = vectorizer.fit_transform(texts)
    # print(dtm)
    # Extract topics from document-term matrix using NMF
    nmf = NMF(n_components=num_topics, random_state=0)
    topic_distributions = nmf.fit_transform(dtm)
    # print(nmf.components_)
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


def get_top_n_indices(topic_distributions, cluster_labels, top_n=10):
    # Get indices of data points that belong to a cluster
    cluster_indices = [np.where(cluster_labels == i)[0] for i in range(np.max(cluster_labels) + 1)]

    # Get top n indices from each cluster
    top_n_indices = []
    for indices in cluster_indices:
        if len(indices) > top_n:
            topic_probs = topic_distributions[indices].sum(axis=1)
            top_n_indices.extend(indices[topic_probs.argsort()[-top_n:][::-1]])

    return top_n_indices  

def plot_dendrogram(linkage_matrix):
    # Plot dendrogram of hierarchical clustering
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.xlabel('Repositories')
    plt.ylabel('Distance')
    plt.title('Cluster Dendrogram')
    plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

def recommend_repos(username):
    # Get user repositories and their descriptions
    user_repos = requests.get(f'https://api.github.com/users/{username}/repos').json()
    user_repo_descriptions = [repo['description'] for repo in user_repos if repo['description']]
    user_repo_languages = [repo['language'] for repo in user_repos if repo['language']]

    # Get top starred repositories and their descriptions
    top_repos_data = get_top_starred_repos()
    top_repo_descriptions = top_repos_data['description'].tolist()
    top_repo_languages = top_repos_data['language'].tolist()

    # Remove np.nan values from descriptions
    top_repo_descriptions = [desc for desc in top_repo_descriptions if isinstance(desc, str)]
    top_repo_languages = [lang for lang in top_repo_languages if isinstance(lang, str)]

    # Concatenate user repository descriptions with top repository descriptions
    texts = user_repo_descriptions + top_repo_descriptions
    languages = user_repo_languages + top_repo_languages

    # Check if there are any repository descriptions available
    if len(texts) == 0:
        print("No repository descriptions available for recommendation")
        return
    # print(texts)
    # Vectorize the text data using TF-IDF
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    # vectorizer.fit(texts)
    # print(vectorizer)
    # Extract topic distributions and language encodings for the text data
    topic_distributions = extract_topic_distributions(texts)
    # print(topic_distributions)
    language_encodings = vectorizer.transform(languages)
    n = max(topic_distributions.shape[0], language_encodings.shape[0])
    topic_distributions_pad = np.pad(topic_distributions, ((0, n - topic_distributions.shape[0]), (0, 0)))
    language_encodings_pad = np.pad(language_encodings.toarray(), ((0, n - language_encodings.shape[0]), (0, 0)))
    # Concatenate topic distributions and language encodings into a single feature matrix
    feature_matrix = np.concatenate([topic_distributions_pad, language_encodings_pad], axis=1)

    # Cluster the repositories using hierarchical clustering
    linkage_matrix = linkage(feature_matrix, method='ward')
    num_clusters = len(np.unique(fcluster(linkage_matrix, 1.5, criterion='distance')))
    st.write("Number of Clusters: ", num_clusters)
    plot_dendrogram(linkage_matrix)
    cluster_labels = fcluster(linkage_matrix, t=1.5,criterion='distance')
    silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
    calinski_harabasz_avg = calinski_harabasz_score(feature_matrix, cluster_labels)
    db_index = davies_bouldin_score(feature_matrix, cluster_labels)

    # st.write("Calinski-Harabasz index:", calinski_harabasz_avg)
    # st.write("silhouette score:"+ str(silhouette_avg))
    # st.write(f"Davies-Bouldin index: {db_index}")


    # Get the indices of the top recommended repositories
    top_n_indices = get_top_n_indices(topic_distributions,cluster_labels, top_n=5)
    # Print the top recommended repositories
    st.write("Top Recommended Repositories:")
    for index in top_n_indices:
        if(index<len(top_repos_data)):
            if cluster_labels[index] == -1:
                continue
            # st.write(f"{top_repos_data.iloc[index]['name']} --- ({top_repos_data.iloc[index]['description']}) --- {top_repos_data.iloc[index]['url']}")
            components.html(""" <style>@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@600;700&display=swap');</style>
                                        <a href="%s" class="data-card" target="_blank" style="display: flex;
                                                                                            flex-direction: column;
                                                                                            max-width: 20.75em;
                                                                                            min-height: 20.75em;
                                                                                            overflow: hidden;
                                                                                            border-radius: 15px;
                                                                                            text-decoration: none;
                                                                                            background: #219F94;
                                                                                            margin: 1em;
                                                                                            padding: 2.75em 2.5em;
                                                                                            box-shadow: 0 1.5em 2.5em -.5em rgba(#000000, .1);
                                                                                            transition: transform .45s ease, background .45s ease">
                
                                        <h3 style="color: white;word-wrap:break-word;
                                        font-size: 2.1em;
                                        font-weight: 600;
                                        line-height: 1;
                                        padding-bottom: .5em;
                                        margin: 0 0 0.142857143em;
                                        border-bottom: 2px solid white;
                                        transition: color .45s ease, border .45s ease;">%s</h3>
                                        <p style="color: white;word-wrap:break-word;
                                                    font-size:1.25em;
                                                    font-weight: 600;
                                                    line-height: 1.8;
                                                    margin: 0 0 1.25em;
                                                    ">%s</p>
                                        <span class="link-text" style="color:white;" >
                                            View 
                                            <svg style="margin-left:0.5em;transition: transform .6s ease;" width="25" height="16" viewBox="0 0 25 16" fill="#FFFFFF" xmlns="http://www.w3.org/2000/svg">
                                        <path fill-rule="evenodd" clip-rule="evenodd" d="M17.8631 0.929124L24.2271 7.29308C24.6176 7.68361 24.6176 8.31677 24.2271 8.7073L17.8631 15.0713C17.4726 15.4618 16.8394 15.4618 16.4489 15.0713C16.0584 14.6807 16.0584 14.0476 16.4489 13.657L21.1058 9.00019H0.47998V7.00019H21.1058L16.4489 2.34334C16.0584 1.95281 16.0584 1.31965 16.4489 0.929124C16.8394 0.538599 17.4726 0.538599 17.8631 0.929124Z" fill="white"/>
                                        </svg>
                                            </span>
                                        </a>"""%(top_repos_data.iloc[index]['url'],top_repos_data.iloc[index]['name'],top_repos_data.iloc[index]['description']),height=500,width=500)

with st.form(key = "form1"):
     Username = st.text_input('Enter GitHub Username')
     submit = st.form_submit_button(label = "Submit")
     if submit:
        recommend_repos(Username)
