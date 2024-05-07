import streamlit as st
import pandas as pd
import json
import re
from streamlit_lottie import st_lottie
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import sent_tokenize



model = Doc2Vec.load('d2v.model')   # <- 모델 불러오기    
test = pd.read_csv('test.csv')

# JSON을 읽어 들이는 함수.
def loadJSON(path):
    f = open(path, 'r')
    res = json.load(f)
    f.close()
    return res

# 로고 Lottie와 타이틀 출력.
col1, col2 = st.columns([1,2])
with col1:
    lottie = loadJSON('Animation  fourprj.json')
    st_lottie(lottie, speed=1, loop=True, width=150, height=150)
with col2:
    ''
    ''
    st.title('유사도 검증')


with st.form(key='myForm', clear_on_submit=False):
    doc_list = st.text_input('문장을 입력해 주세요')
    similarity = st.slider('유사도 비율',0.0, 1.0, 0.5, 0.1)
    submit = st.form_submit_button('실행')


doc_list = sent_tokenize(doc_list)
doc_list = [x.lower() for x in doc_list]

t = []
for a_sentence in doc_list:
    a_sentence = re.sub(r'\W',' ',a_sentence)            # 특수 문자는 space로 대체.
    a_sentence = re.sub(r'\d', '', a_sentence)
    a_sentence = re.sub(r'\s+',' ',a_sentence)           # 잉여 space 제거.
    t.append(a_sentence)
while ' ' in t[:]:
    t.remove(' ')

t = pd.DataFrame({'sentence':t})

model.random.seed(9999)

for i in range(len(t)):
    sent = t['sentence'][i]
    sent2 = sent.split(' ')

    inferred_vector = model.infer_vector(sent2)
    return_docs = model.dv.most_similar(positive=[inferred_vector],topn=5)

    st.write(f'<span style="color:skyblue">{sent}</span>', unsafe_allow_html=True)

    for rd in return_docs:
        for des in test[test['num'] == rd[0]]['sentence']:
            if rd[1] > similarity:
                
                st.write(rd[0],rd[1],'\n',des,'\n')
    st.write('='*80)

# model.random.seed(9999)
# doc_list = doc_list.split(' ')

# inferred_vector = model.infer_vector(doc_list)
# return_docs = model.dv.most_similar(positive=[inferred_vector],topn=5)

# for rd in return_docs:
#     for des in test[test['num'] == rd[0]]['sentence']:
#         if rd[1] > similarity:
#             st.write(rd[0],rd[1],'\n',des,'\n')

   



   



