import streamlit as st
import pandas as pd
import pickle

# 加载模型
model = pickle.load(open('women_depression_rf_model.pkl', 'rb'))

# 预测函数
def make_prediction(model, input_features):
    input_features_df = pd.DataFrame([input_features], columns=[
        'Age', 'Major life events', 'Negative thoughts/behaviors', 'Number of suicide attempts',
        'IPAQ level', 'BMI', 'Sleep quality', 'Perceived stress', 'Hopelessness',
        'Loneliness', 'Resilience', 'Alexithymia', 'Problem-focused coping',
        'Emotion-focused coping', 'Self-esteem', 'Rumination', 'Emotion regulation',
        'Borderline personality', 'Care', 'Overprotection'
    ])
    prediction = model.predict(input_features_df)
    return prediction[0]

# Streamlit 应用界面
st.title('Predicting Depression in Adolescent Girls with Non-Suicidal Self-Injury')

# 创建输入控件
age = st.number_input('Age', min_value=10, max_value=18, step=1)
major_life_events = st.number_input('Major life events', min_value=0, max_value=1, step=1)
negative_thoughts_behaviors = st.number_input('Negative thoughts/behaviors', min_value=0, max_value=100, step=1)
number_suicide_attempts = st.number_input('Number of suicide attempts', min_value=0, max_value=2, step=1)
ipaq_level = st.number_input('IPAQ level', min_value=1, max_value=3, step=1)
bmi = st.number_input('BMI', min_value=13, max_value=50, step=1)
sleep_quality = st.number_input('Sleep quality', min_value=0, max_value=21, step=1)
perceived_stress = st.number_input('Perceived stress', min_value=4, max_value=20, step=1)
hopelessness = st.number_input('Hopelessness', min_value=20, max_value=100, step=1)
loneliness = st.number_input('Loneliness', min_value=23, max_value=80, step=1)
resilience = st.number_input('Resilience', min_value=0, max_value=40, step=1)
alexithymia = st.number_input('Alexithymia', min_value=26, max_value=97, step=1)
problem_focused_coping = st.number_input('Problem-focused coping', min_value=20, max_value=80, step=1)
emotion_focused_coping = st.number_input('Emotion-focused coping', min_value=17, max_value=66, step=1)
self_esteem = st.number_input('Self-esteem', min_value=10, max_value=40, step=1)
rumination = st.number_input('Rumination', min_value=22, max_value=88, step=1)
emotion_regulation = st.number_input('Emotion regulation', min_value=10, max_value=50, step=1)
borderline_personality = st.number_input('Borderline personality', min_value=25, max_value=120, step=1)
care = st.number_input('Care', min_value=0, max_value=36, step=1)
overprotection = st.number_input('Overprotection', min_value=0, max_value=36, step=1)

# 当用户点击预测按钮时，显示预测结果
if st.button('预测'):
    # 创建输入特征列表
    input_features = [
        age, major_life_events, negative_thoughts_behaviors, number_suicide_attempts,
        ipaq_level, bmi, sleep_quality, perceived_stress, hopelessness, loneliness,
        resilience, alexithymia, problem_focused_coping, emotion_focused_coping,
        self_esteem, rumination, emotion_regulation, borderline_personality,
        care, overprotection
    ]
    
    # 进行预测
    prediction = make_prediction(model, input_features)
    st.write('抑郁预测结果:', prediction)
