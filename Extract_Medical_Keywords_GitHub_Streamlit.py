# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:23:02 2025

@author: user
"""



# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:06:55 2024

@author: user
"""

import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import xml.etree.ElementTree as ET
import requests
import os
import pandas as pd
from collections import Counter
import plotly.express as px

# 確認是否已經下載所需的NLTK資源，否則進行下載
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_punkt_stopwords():
    nltk.download('punkt')
    nltk.download('stopwords')

load_punkt_stopwords()

def preprocess_text(text):
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def download_mesh_data():
    url = "https://www.dropbox.com/scl/fi/vx7gcihrli5kcj331psbh/desc2025.xml?rlkey=rzfb0cq34odikah9kzq8r64u2&st=kfxsy44r&dl=1"
    file_path = "desc2025.xml"
    if not os.path.exists(file_path):
        st.write(f"Downloading MeSH data from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as file:
                file.write(response.content)
            st.write("下載完成")
            with open(file_path, 'r', encoding='utf-8') as file:
                first_lines = ''.join([file.readline() for _ in range(10)])
        except requests.exceptions.RequestException as e:
            st.error(f"下載MeSH文件時發生錯誤: {e}")
            return None
    return file_path

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_mesh_terms(file_path):
    if file_path and os.path.exists(file_path):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            mesh_terms = set()
            for descriptor in root.findall(".//DescriptorRecord/DescriptorName/String"):
                mesh_terms.add(descriptor.text.lower())
            return mesh_terms
        except ET.ParseError as e:
            st.error(f"XML解析錯誤: {e}")
            return set()
        except Exception as e:
            st.error(f"讀取或解析MeSH文件時發生錯誤: {e}")
            return set()
    else:
        st.error("MeSH文件不存在或路徑錯誤")
        return set()

mesh_file_path = download_mesh_data()
mesh_terms = load_mesh_terms(mesh_file_path)

st.title("醫學關鍵字提取與排序 (根據 TF-IDF 與 MeSH 詞彙集)")

uploaded_files = st.file_uploader("上傳你的文本或PDF文件", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
        texts.append(text)

    processed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    all_keywords_data = []

    for i, text in enumerate(processed_texts):
        st.subheader(f"Text {i+1}:")
        tfidf_scores = list(zip(feature_names, tfidf_matrix[i].toarray()[0]))
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        wordlist = text.split()
        counter = Counter(word for word in wordlist if word in mesh_terms)
        tfidf_map = {keyword: score for keyword, score in sorted_scores if score > 0 and keyword in mesh_terms}

        for keyword in counter:
            all_keywords_data.append({
                "Text Index": i + 1,
                "Keyword": keyword,
                "Count": counter[keyword],
                "TF-IDF": tfidf_map.get(keyword, 0)
            })

        for keyword, score in sorted_scores:
            if score > 0 and keyword in mesh_terms:
                st.write(f"Medical Keyword: {keyword}, Score: {score:.4f}")
        st.write("\n")

    if all_keywords_data:
        df = pd.DataFrame(all_keywords_data)
        # 以 Count 由大到小排序，若 Count 相同以 TF-IDF 由大到小
        df = df.sort_values(by=["Count", "TF-IDF"], ascending=[False, False])
        st.dataframe(df)

        # 匯出CSV（已排序）
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="下載關鍵字統計結果 (CSV)",
            data=csv,
            file_name='medical_keywords_sorted.csv',
            mime='text/csv'
        )

        # 畫圖：出現次數 Top 20
        st.subheader("依關鍵字出現次數排序 (Top 20)")
        top_counts = df.groupby("Keyword")["Count"].sum().sort_values(ascending=False).head(20)
        st.bar_chart(top_counts)

        # 畫圖：TF-IDF Top 20
        st.subheader("依 TF-IDF 分數排序 (Top 20)")
        top_tfidf = df.groupby("Keyword")["TF-IDF"].sum().sort_values(ascending=False).head(20)
        st.bar_chart(top_tfidf)

        # 散佈圖（橫軸次數，縱軸 TF-IDF，點皆按最大排序）
        st.subheader("關鍵字出現次數 vs. TF-IDF（散佈圖）")
        grouped = df.groupby("Keyword").agg({'Count': 'sum', 'TF-IDF': 'sum'}).reset_index()
        grouped = grouped.sort_values(by=["Count", "TF-IDF"], ascending=[False, False])
        fig = px.scatter(grouped, x='Count', y='TF-IDF', text='Keyword', title='Count vs. TF-IDF')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig)
else:
    st.write("請上傳一個或多個文本或PDF文件。")
