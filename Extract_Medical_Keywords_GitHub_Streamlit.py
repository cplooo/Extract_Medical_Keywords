# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:50:11 2025

@author: user
"""


# -*- coding: utf-8 -*-
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
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_punkt_stopwords():
    nltk.download('punk_tab')
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




# ## 直接從 National Library of Medicine 下載:
# @st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
# def download_mesh_data():
#     # url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2023.xml"
#     url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
           
#     file_path = "desc2025.xml"
#     if not os.path.exists(file_path):
#         st.write(f"Downloading MeSH data from {url}...")
#         try:
#             response = requests.get(url)
#             response.raise_for_status()  # 確保請求成功

#             # 將文件寫入本地文件系統
#             with open(file_path, 'wb') as file:
#                 file.write(response.content)
#             st.write("下載完成")

#             # 檢查文件內容是否正確
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 first_lines = ''.join([file.readline() for _ in range(10)])
#                 # st.write("desc2024.xml 文件頭部內容:")
#                 # st.text(first_lines)

#         except requests.exceptions.RequestException as e:
#             st.error(f"下載MeSH文件時發生錯誤: {e}")
#             return None
    
#     return file_path



## 從 dropbox下載:
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
    filenames = []
    for uploaded_file in uploaded_files:
        filenames.append(uploaded_file.name)
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
        texts.append(text)

    # 預處理
    processed_texts = [preprocess_text(text) for text in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    for i, processed_text in enumerate(processed_texts):
        st.header(f"📄 文件 {i+1}: {filenames[i]}")
        tfidf_scores = list(zip(feature_names, tfidf_matrix[i].toarray()[0]))
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        wordlist = processed_text.split()
        counter = Counter(word for word in wordlist if word in mesh_terms)
        tfidf_map = {keyword: score for keyword, score in sorted_scores if score > 0 and keyword in mesh_terms}
        keywords_data = []

        for keyword in counter:
            keywords_data.append({
                "Keyword": keyword,
                "Count": counter[keyword],
                "TF-IDF": tfidf_map.get(keyword, 0)
            })

        # 資料表 & 下載按鈕
        if keywords_data:
            df = pd.DataFrame(keywords_data)
            # 依 Count, TF-IDF 由大到小排序
            df = df.sort_values(by=["Count", "TF-IDF"], ascending=[False, False])
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下載本檔關鍵字統計結果 (CSV)",
                data=csv,
                file_name=f'medical_keywords_{i+1}_{filenames[i]}.csv',
                mime='text/csv'
            )




            # 條狀圖：依關鍵字出現次數排序 (Top 20)
            st.subheader("依關鍵字出現次數排序 (Top 20)")
            top_counts = df[["Keyword", "Count"]].drop_duplicates().sort_values("Count", ascending=False).head(20)
            fig_count = go.Figure(go.Bar(
                x=top_counts["Count"][::-1],  # 反轉讓最大值在上方
                y=top_counts["Keyword"][::-1],
                orientation='h'
            ))
            fig_count.update_layout(
                xaxis_title="Count",
                yaxis_title="Keyword",
                height=500
            )
            st.plotly_chart(fig_count, use_container_width=True)
            
            # 條狀圖：依 TF-IDF 分數排序 (Top 20)
            st.subheader("依 TF-IDF 分數排序 (Top 20)")
            top_tfidf = df[["Keyword", "TF-IDF"]].drop_duplicates().sort_values("TF-IDF", ascending=False).head(20)
            fig_tfidf = go.Figure(go.Bar(
                x=top_tfidf["TF-IDF"][::-1],  # 反轉讓最大值在上方
                y=top_tfidf["Keyword"][::-1],
                orientation='h'
            ))
            fig_tfidf.update_layout(
                xaxis_title="TF-IDF",
                yaxis_title="Keyword",
                height=500
            )
            st.plotly_chart(fig_tfidf, use_container_width=True)
            
            
            

            st.subheader("關鍵字出現次數 vs. TF-IDF（散佈圖）")
            grouped = df.groupby("Keyword").agg({'Count': 'sum', 'TF-IDF': 'sum'}).reset_index()
            grouped = grouped.sort_values(by=["Count", "TF-IDF"], ascending=[False, False])
            fig = px.scatter(grouped, x='Count', y='TF-IDF', text='Keyword', title='Count vs. TF-IDF')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("本檔案沒有找到任何 MeSH 關鍵字")
else:
    st.write("請上傳一個或多個文本或PDF文件。")
