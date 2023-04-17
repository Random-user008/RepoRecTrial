import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from streamlit_card import card
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="GitHub Project Recommendation System",
    page_icon="GitHub-icon.png",
)
components.html("""
     <style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative&display=swap');</style>
    <h1 style="color:ZBlack;font-family: 'Cinzel Decorative', cursive;
font-size:30px">Enter the details to Recommend Projects</h1>    """,height=100,width=700)
# Username = st.text_input('Enter GitHub Username')
# st.write("UserName Entered is: ",Username)
# time.sleep(8)
# st.write("LOADING... ")


# Load data
df = pd.DataFrame()

# This is the standar csv, with the gravatar images for each repo
df = pd.read_csv("https://raw.githubusercontent.com/kaayush71/shopping-cart-CSV/main/TopStaredRepositories.csv")
df.set_index(['Repository Name'])

# st.write(df.head(2))
## Cleaning data
# We fill the emptpy URL cells
df['Url'] = "http://github.com/" + df['Username'] + "/" + df['Repository Name']
# We add a final comma character for the tag string, it will be usefull when we tokenize
df['Tags'].fillna("", inplace=True)
df['Tags'] = df['Tags'] + ","

# We do not want uppercase on any label
df['Language'] = df.loc[:, 'Language'].str.lower()
# Copy a backup variable, so we can change our main dataframe
df_backup = df.copy(deep=True)
# st.write(df.head(2))
mergedlist = []
for i in df['Tags'].dropna().str.split(","):
    mergedlist.extend(i)
tags = sorted(set(mergedlist))
# Encode languages in single column
just_dummies = pd.get_dummies(df['Language'])
for column in just_dummies.columns:
    if column not in df.columns:
        df[column] = just_dummies[column]
# st.write(df.head(2))
for tag in tags:
    if tag not in df.columns:
        df[tag] = 0
    try:
        if len(tag) > 4:
            df.loc[df['Repository Name'].str.contains(tag), tag] += 1
            df.loc[df['Description'].str.contains(tag), tag] += 1
        df.loc[df['Tags'].str.contains(tag + ","), tag] += 1
    except Exception:
        pass
# Remove columns not needed
df.set_index(['Repository Name'])
COLUMNS_TO_REMOVE_LIST = ['', 'Username', 'Repository Name', 'Description',
                          'Last Update Date', 'Language', 'Number of Stars', 'Tags', 'Url', 'Gravatar', 'Unnamed: 0']
# Stop words: links to (https://github)
# RAGE_TAGS_LIST = ['github', 'algorithms', 'learn', 'learning', 'http', 'https']

# for column in COLUMNS_TO_REMOVE_LIST + RAGE_TAGS_LIST:
#     try:
#         del df[column]
#     except Exception:
#         pass

df.columns = df.columns.str.lower()

print("Our final label matrix for repo list is")
# st.write(df.head(2))

# Apply LDA to extract topics from

text_data = df_backup['Description'].astype(str)
tokens = [simple_preprocess(str(desc), deacc=True) for desc in text_data]
tokens = [[word for word in doc if word not in STOPWORDS] for doc in tokens]

dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(doc) for doc in tokens]

lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=100,
                            chunksize=1000, passes=50, iterations=100, per_word_topics=True)

# Extracting topics
topics = lda_model.show_topics(formatted=False)
topics = [(topic[0], [t[0] for t in topic[1]]) for topic in topics]

# Displaying the topics
st.write("These are the extracted topics from the repository descriptions:")
for topic in topics:
    st.write(f"Topic {topic[0]}: {', '.join(topic[1])}")

# Create a function to recommend projects
def recommend_projects(input_tags, num_projects=5):
    """
    Recommend projects based on input tags.

    Parameters:
    -----------
    input_tags: list
        List of input tags.
    num_projects: int
        Number of projects to recommend.

    Returns:
    --------
    recommended_projects: pandas.DataFrame
        Dataframe containing the recommended projects.
    """
    input_tags = [tag.lower() for tag in input_tags]
    tag_matrix = df[input_tags]
    tag_count = tag_matrix.sum(axis=1)
    project_scores = tag_count.sort_values(ascending=False)
    recommended_projects = df_backup.loc[project_scores.index[:num_projects]]
    recommended_projects['Url'] = "http://github.com/" + recommended_projects['Username'] + "/" + recommended_projects[
        'Repository Name']
    recommended_projects = recommended_projects[['Repository Name', 'Url', 'Number of Stars', 'Language', 'Tags',
                                                 'Description']]
    return recommended_projects

# Display the recommendation form
components.html("""
     <style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap');</style>
    <h2 style="color:ZBlack;font-family: 'Roboto Mono', monospace;
font-size:20px">Fill out the form to get project recommendations:</h2>    """,height=60,width=700)

input_tags = st.text_input('Enter tags (comma separated)')
num_projects = st.slider('Number of projects to recommend', min_value=1, max_value=20, value=5)
if st.button('Recommend'):
    input_tags = input_tags.split(',')
    recommended_projects = recommend_projects(input_tags, num_projects)
    if recommended_projects.empty:
        st.write('No projects found with these tags')
    else:
        st.write(recommended_projects)

