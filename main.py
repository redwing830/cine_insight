import numpy as np
import pandas as pd
import streamlit as st
import re
import csv
import openai
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from datetime import date as d_date

st.set_page_config(layout="wide")

openai.api_key = st.secrets.OPENAI_TOKEN
openai_model_version = "gpt-3.5-turbo"

with open('./.streamlit/wave.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

pickle_file = './data/dictionary_data.pkl'

with open(pickle_file, 'rb') as file:
    dictionary_data = pickle.load(file)

def read_unique_nationalities(file_path):
    nationalities = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)

        for row in reader:
            nationality = row[0] 
            nationalities.append(nationality)
    return nationalities

def generate_prompt(title, genres, director, cast, screenplay, original_work,
                    runtime, rating, num_screens, nationality,
                    series_value,total_data):
    prompt = f"""
            제목: {title}
            장르: {''.join(genres)}
            감독: {director}
            주연: {''.join(cast)}
            각본: {screenplay}
            원작: {original_work}
            런타임: {runtime}
            등급: {rating}
            스크린수: {num_screens}
            시리즈의 여부: {series_value}
            영화의 국적: {nationality}
            머신러닝 으로 예측한 관객수: {total_data}
    ------------------------------------------
    이 모델은 아직 개봉하지 않은 영화에 대해 사용자로부터 위와 같은 데이터를 입력받은 뒤 머신러닝을 이용해 한국 극장가에서의 관객 수를 예측합니다.
    사용자가 입력한 영화 제목과 예측된 관객 수를 언급하는 것을 시작으로 왜 이 영화가 그러한 관객 수로 예측되었는지 그 이유를 설명해주세요.
    당신의 설명은 누가 들어도 합리적이고 타당해야 합니다. 사실 여부가 불분명한 부분은 언급하지 말고 특히 없는 사실을 지어내지 마세요.
    관객수가 적게 예측됐더라도 가급적 긍정적인 메시지를 주도록 노력하세요.
    
    ------------------------------------------
    영화의 국적: 열거된 여러 나라 중 하나입니다.
    시리즈의 여부: 1은 같은 시리즈물이 예전에 1편 이상 개봉됐습니다.

    """
    return prompt.strip()


def request_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model=openai_model_version,
        messages=[
            {"role": "system", "content": "당신은 유용한 도우미입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

def normalize_names(name):
    names = re.split(r',\s*', name)
    sorted_names = sorted(names)
    return ', '.join(sorted_names)

def get_avg_audience_by_person_and_date(person, date, role):
    names = re.split(r',\s*', person)
    if role == 'director':
        names = [normalize_names(name) for name in names]

    def name_match(dictionary):
        for name in names:
            if name not in dictionary:
                return False
        return True

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    if role == 'actor':
        matching_dicts = [value for key, value in actor_avg_audience.items() if name_match(key)]
    elif role == 'director':
        matching_dicts = [value for key, value in director_avg_audience.items() if name_match(key)]
    elif role == 'scriptwriter':
        matching_dicts = [value for key, value in scriptwriter_avg_audience.items() if name_match(key)]
    elif role == 'writer':
        matching_dicts = [value for key, value in writer_avg_audience.items() if name_match(key)]
    else:
        return 0

    if not matching_dicts:
        return 0

    max_value = -1
    for person_dict in matching_dicts:
        closest_date = None
        for release_date in person_dict.keys():
            if release_date <= pd.Timestamp(date) and (closest_date is None or release_date > closest_date):
                closest_date = release_date

        if closest_date is not None:
            value_str = person_dict[closest_date]
            if is_number(value_str):
                value = int(value_str)
                max_value = max(max_value, value)

    if max_value > -1:
        return max_value
    else:
        return 0

def get_highest_avg_audience(date, role, limit=10):
    if role == 'actor':
        data_dict = actor_avg_audience
    elif role == 'director':
        data_dict = director_avg_audience
    elif role == 'scriptwriter':
        data_dict = scriptwriter_avg_audience
    elif role == 'writer':
        data_dict = writer_avg_audience
    else:
        return []

    avg_audience_dict = {}
    for person, data in data_dict.items():
        max_date = d_date(1900, 1, 1)
        avg_audience = 0.0
        for date_key, audience in data.items():
            curr_date = pd.to_datetime(date_key)
            if curr_date.date() <= date and curr_date.date() > max_date:
                max_date = curr_date.date()
                avg_audience = float(audience) if audience != '' else 0.0
        avg_audience_dict[person] = avg_audience

    avg_audience = [(person, avg_audience) for person, avg_audience in avg_audience_dict.items()]
    avg_audience.sort(key=lambda x: x[1], reverse=True)

    return avg_audience[:limit]


def genre_to_onehot(input_genres):
    genre_columns = ['genre_SF', 'genre_가족', 'genre_공연', 'genre_공포',
                     'genre_기타', 'genre_다큐멘터리', 'genre_드라마', 'genre_로맨스', 'genre_뮤지컬',
                     'genre_미스터리', 'genre_범죄', 'genre_사극', 'genre_서부극', 'genre_성인물',
                     'genre_스릴러', 'genre_애니메이션', 'genre_액션', 'genre_어드벤처', 'genre_전쟁',
                     'genre_코미디', 'genre_판타지']

    genre_dict = {genre: [0] for genre in genre_columns}

    for genre in input_genres: 
        genre = 'genre_' + genre.strip() 
        if genre in genre_dict:
            genre_dict[genre] = [1] 

    return genre_dict

def draw_graph(predicted_value):
    font_path = './data/NanumGothic.otf'
    fontprop = fm.FontProperties(fname=font_path, size=15)

    df = pd.read_csv('./data/preprocessed.csv')
    predicted_audience = float(predicted_value.replace(',', ''))

    fig, ax = plt.subplots()
    fig.set_facecolor('#7972d0')
    ax.set_facecolor('#7972d0')
    sns.kdeplot(data=df, x="audience", bw_adjust=1.5, color='b', ax=ax, fill=True)
    ax.set_xlabel('관객수', fontproperties=fontprop, color='white')
    ax.set_ylabel('확률밀도', fontproperties=fontprop, color='white')
    ax.set_xscale('log')

    log_audience = np.log10(df['audience'])
    log_predicted_audience = np.log10(predicted_audience)
    closest_index = (np.abs(log_audience - log_predicted_audience)).idxmin()
    percentile = (closest_index / (len(df) - 1)) * 100

    ax.axvline(predicted_audience, color='red', linestyle='--')
    x_coordinate = 10 ** (log_predicted_audience + 0.2)
    ax.text(x_coordinate, ax.get_ylim()[1] / 2,
            f"예상 관객수: {predicted_audience:,.0f}명\n  (전체 상위 {percentile:.2f}%)",
            color='yellow', fontproperties=fontprop)

    def human_readable_number(x, pos):
        if x >= 1e6:
            return f"{x * 1e-6:.0f}M"
        elif x >= 1e3:
            return f"{x * 1e-3:.0f}k"
        else:
            return f"{x:.0f}"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_readable_number))
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='x', labelrotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    st.pyplot(fig)

genres_list = ['스릴러', '액션', 'SF', '가족', '공연', '공포', '기타', '다큐멘터리', '드라마', '로맨스', '뮤지컬', '미스터리', '범죄', '사극', '서부극',
                   '성인물', '애니메이션', '어드벤처', '전쟁', '코미디', '판타지']

genres_list = sorted(genres_list, reverse=False)

nationality_list = ['기타', '미국_캐나다', '유럽', '일본', '중국_대만_홍콩', '한국']
nationality_list = sorted(nationality_list, reverse=False)

rating_list = ['전체관람가', '12세관람가', '15세관람가', '청소년관람불가']

actor_avg_audience = dictionary_data['actor_avg_audience']
director_avg_audience = dictionary_data['director_avg_audience']
scriptwriter_avg_audience = dictionary_data['scriptwriter_avg_audience']
writer_avg_audience = dictionary_data['writer_avg_audience']

actor_avg_audience = dict(sorted(actor_avg_audience.items(), key=lambda item: item[0]))
director_avg_audience = dict(sorted(director_avg_audience.items(), key=lambda item: item[0]))
scriptwriter_avg_audience = dict(sorted(scriptwriter_avg_audience.items(), key=lambda item: item[0]))
writer_avg_audience = dict(sorted(writer_avg_audience.items(), key=lambda item: item[0]))

actor_avg_audience.pop('nan', None)

def nan_to_select_none_and_sort(dictionary):
    if 'nan' in dictionary:
        dictionary['선택 안함'] = dictionary.pop('nan')

    sorted_dictionary = dict(sorted(dictionary.items(), key=lambda x: x[0]))

    if '선택 안함' in sorted_dictionary:
        sorted_dictionary = {'선택 안함': sorted_dictionary['선택 안함'], **sorted_dictionary}
    
    return sorted_dictionary

director_avg_audience = nan_to_select_none_and_sort(director_avg_audience)
scriptwriter_avg_audience = nan_to_select_none_and_sort(scriptwriter_avg_audience)
writer_avg_audience = nan_to_select_none_and_sort(writer_avg_audience)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    tab1, tab2 = st.tabs(["관객수 예측", "영화인 흥행력"])
    with tab1:
        title = st.text_input("영화 제목을 입력하세요.")
        genres = st.multiselect("장르를 선택하세요. (최대 3개)", genres_list, max_selections=3)

        director_list = list(director_avg_audience.keys())
        default_index = director_list.index('nan') if 'nan' in director_list else 0
        director = st.selectbox("감독을 선택하세요.", director_list, index=default_index)

        actor_list = list(actor_avg_audience.keys())
        cast = st.multiselect("주연 배우를 선택하세요. (최대 3명)", actor_list, max_selections=3)
        
        col1, col2 = st.columns(2)
        with col1:
            scriptwriter_list = list(scriptwriter_avg_audience.keys())
            default_index = scriptwriter_list.index('nan') if 'nan' in scriptwriter_list else 0
            screenplay = st.selectbox("각본 작가를 선택하세요.", scriptwriter_list, index=default_index)

        with col2:
            writer_list = list(writer_avg_audience.keys())
            default_index = writer_list.index('nan') if 'nan' in writer_list else 0
            original_work = st.selectbox("원작자를 선택하세요.", writer_list, index=default_index)

        col1, col2 = st.columns(2)

        with col1:
            runtime = st.number_input("상영 시간을 입력하세요 (분)", min_value=1, value=100)

            if runtime < 90:
                runtime_category = 1
            elif runtime <= 110:
                runtime_category = 2
            else:
                runtime_category = 3

        with col2:
            num_screens = st.number_input("상영 스크린 수를 입력하세요.", min_value=1, value=50)

        col1, col2 = st.columns(2)

        with col1:
            rating = st.selectbox("관람 등급을 선택하세요.", rating_list)

        with col2:
            nationality = st.selectbox("영화의 국적을 선택하세요.", nationality_list, index=5)

        col1, col2 = st.columns(2)

        with col1:
            is_series = st.checkbox("이 영화는 시리즈물입니까?")
            series_value = 1 if is_series else 0

        with col2:
            is_ai = st.checkbox("AI 분석결과를 보시겠습니까?")
            ai_value = 1 if is_ai else 0

        st.markdown('***')
 
        predict_button = st.button(label="영화 관객수 예측")


    with tab2:
        with st.container():
            choice_dict = {
                '배우': 'actor_avg_audience',
                '감독': 'director_avg_audience',
                '각본': 'scriptwriter_avg_audience',
                '원작': 'writer_avg_audience'
            }
            
            choice = st.selectbox("직종을 선택하세요.", list(choice_dict.keys()), key='choice1')

            def process_person_data(person_data):
                for person_name in list(person_data):
                    if ', ' in person_name:
                        names = person_name.split(', ')
                        names.sort()
                        person_data[', '.join(names)] = person_data.pop(person_name)
                return person_data

            selected_data = process_person_data(dictionary_data[choice_dict[choice]])

            person_list = list(selected_data.keys())
            person_list = sorted(person_list, key=lambda x: re.sub(r'[^가-힣a-zA-Z]', '', x))
            person_list = [
                person for person in person_list
                if person != 'nan' and selected_data[person] != 0
            ]

            default_index = 0 if person_list else None
            person_p2 = st.selectbox("영화인을 선택하세요.", person_list, index=default_index, key='person_p2')

            selected_date = st.date_input("날짜를 선택하세요.")
            
            choice_eng = {
                '배우': 'actor',
                '감독': 'director',
                '각본': 'scriptwriter',
                '원작': 'writer'
            }[choice]

            avg_audience = get_avg_audience_by_person_and_date(person_p2, selected_date, choice_eng)
            
            if isinstance(avg_audience, str):
                formatted_avg_audience = avg_audience
            else:
                formatted_avg_audience = "{:,.0f}".format(avg_audience)

            st.write(f"{person_p2}의 선택 시점 평균 관객수는 {formatted_avg_audience} 명입니다.")


        st.markdown("---")

        with st.container():
            st.subheader('영화인 흥행력 TOP 10')

            choice_p3_dict = {'배우': 'actor_avg_audience', '감독': 'director_avg_audience', '각본': 'scriptwriter_avg_audience', '원작': 'writer_avg_audience'}
            choice_p3 = st.selectbox("직종을 선택하세요.", list(choice_p3_dict.keys()), key='choice2')
            selected_data_p3 = dictionary_data[choice_p3_dict[choice_p3]]
            selected_date_p3 = st.date_input("날짜를 선택하세요.", key='selected_date_p3')
            choice_p3_eng = {'배우': 'actor', '감독': 'director', '각본': 'scriptwriter', '원작': 'writer'}[choice_p3]
            result = get_highest_avg_audience(selected_date_p3, choice_p3_eng, limit=10)

            for idx, (person, avg_audience) in enumerate(result, start=1):
                formatted_avg_audience = "{:,.0f}".format(avg_audience)
                line = f"{idx}. {person}"
                line = f"<div style='display: flex; justify-content: space-between; margin-left: 10%; margin-right: 10%;'>{line}<div style='text-align: right;'>{formatted_avg_audience} 명</div></div>"
                st.markdown(line, unsafe_allow_html=True)

st.markdown(
    """
    <link href='https://fonts.googleapis.com/css2?family=Tangerine:wght@700&display=swap' rel='stylesheet'>
    <style>
    h1 {
        text-align: center;
        font-size: 70px;
        font-family: 'Tangerine', cursive;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>CineInsight 📽</h1>", unsafe_allow_html=True)


st.write()
placeholder = st.empty()

with placeholder.container():
    st.write()
    youtube_embed_code = '''
    <div style="display: flex; justify-content: center;">
        <div style="width: 70%;">
            <div style="position: relative; padding-bottom: 56.25%; /* 16:9 비율에 맞춰 조정 */">
                <iframe src="https://www.youtube.com/embed/jmF0edcnw4w" frameborder="0" allowfullscreen
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
            </div>
        </div>
    </div>
    '''
    st.markdown(youtube_embed_code, unsafe_allow_html=True)
    st.markdown('\n\n')
    st.markdown("<h2 style='text-align: center;'>🍿🥤 영화 흥행 성적을 예측해 드립니다 🥤🍿</h2>", unsafe_allow_html=True)

if predict_button:
    if not genres:
        st.warning("적어도 하나 이상의 장르를 선택해야 합니다.")
        st.stop()

    genres_str = ", ".join(genres)
    cast_data = ", ".join(cast)

    cast_list = [name.strip() for name in cast] 

    total_actor_avg_audience = []
    for name in cast_list:
        avg_audience = get_avg_audience_by_person_and_date(name, '2023-07-10', 'actor')
        if isinstance(avg_audience, (int, float)):
            total_actor_avg_audience.append(avg_audience)
        elif avg_audience == '':
            total_actor_avg_audience.append(0)
        else:
            st.warning(f"배우 '{name}'의 평균 관객수가 잘못된 형식입니다.")
            st.write(f"배우 '{name}'의 평균 관객수: {avg_audience}")

    total_actor_avg_audience_sum = sum(total_actor_avg_audience)
    if total_actor_avg_audience_sum == 0:
        total_actor_avg_audience_sum = 9815 

    director_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'director')
    if director_avg_audience == 0 or director_avg_audience == '':
        director_avg_audience = 9815

    scriptwriter_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'scriptwriter')
    if scriptwriter_avg_audience == 0 or scriptwriter_avg_audience == '':
        scriptwriter_avg_audience = 9815

    writer_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'writer')
    if writer_avg_audience == '':
        writer_avg_audience = 0

    with open('./data/movie_scaler.pkl', 'rb') as file:
        scaler_dict = pickle.load(file)

    scaled_actor_changed = scaler_dict['actor_changed'].transform(
        np.array([total_actor_avg_audience_sum]).reshape(-1, 1))
    new_data = pd.DataFrame({
        'actor_changed': [scaled_actor_changed],
        'screens': [num_screens],
        'director_changed': [director_avg_audience],
        'scriptwriter_changed': [scriptwriter_avg_audience],
        'writer_changed': [writer_avg_audience],
    }).astype(float)

    for column in ['screens', 'director_changed', 'scriptwriter_changed', 'writer_changed']:
        new_data[column] = scaler_dict[column].transform(new_data[[column]])

    genre_data = pd.DataFrame(genre_to_onehot(genres))
    corona_value = 0

    new_data = pd.concat([new_data, genre_data], axis=1)
    total_data = pd.DataFrame(genre_data)

    extra_data = {
        'running_time': [runtime],
        'series_0': [1 if series_value == 0 else 0],
        'series_1': [1 if series_value == 1 else 0],
        '기타': [1 if nationality == '기타' else 0],
        '미국_캐나다': [1 if nationality == '미국_캐나다' else 0],
        '유럽': [1 if nationality == '유럽' else 0],
        '일본': [1 if nationality == '일본' else 0],
        '중국_대만_홍콩': [1 if nationality == '중국_대만_홍콩' else 0],
        '한국': [1 if nationality == '한국' else 0],
        '12세관람가': [1 if rating == '12세관람가' else 0],
        '15세관람가': [1 if rating == '15세관람가' else 0],
        '전체관람가': [1 if rating == '전체관람가' else 0],
        '청소년관람불가': [1 if rating == '청소년관람불가' else 0]
    }
    for key, value in extra_data.items():
        new_data[key] = value

    total_data = pd.DataFrame(new_data)

    ordered_columns = ['running_time', 'screens', 'actor_changed', 'director_changed', 'scriptwriter_changed',
                    'writer_changed', 'genre_SF', 'genre_가족', 'genre_공연',
                    'genre_공포',
                    'genre_기타', 'genre_다큐멘터리', 'genre_드라마', 'genre_로맨스', 'genre_뮤지컬', 'genre_미스터리',
                    'genre_범죄',
                    'genre_사극', 'genre_서부극', 'genre_성인물', 'genre_스릴러', 'genre_애니메이션', 'genre_액션',
                    'genre_어드벤처',
                    'genre_전쟁', 'genre_코미디', 'genre_판타지', 'series_0', 'series_1', '기타', '미국_캐나다', '유럽', '일본',
                    '중국_대만_홍콩', '한국', '12세관람가', '15세관람가', '전체관람가', '청소년관람불가']

    total_data = total_data.reindex(columns=ordered_columns)

    with open('./data/model.pkl', 'rb') as file:
        model = pickle.load(file)

    predicted = model.predict(total_data)

    predicted_int = int(predicted)

    formatted_predicted = "{:,}".format(int(predicted[0]))
    placeholder.empty()
    if title:
        title_text = f"영화 &lt;{title}&gt;의"
    else:
        title_text = "이 영화의"

    st.markdown(f"<h2 style='text-align: center;'>{title_text} 예상 관객 수는 {formatted_predicted} 명입니다.</h2>", unsafe_allow_html=True)
    st.write()

    if ai_value == 1:
        col1, _, col3 = st.columns([8, 1, 8])
        with col1:
            st.markdown("\n\n")
            st.markdown("\n\n")
            st.markdown(f"<h5 style='text-align: center;'>역대 개봉영화 대비 예상 흥행률</h5>", unsafe_allow_html=True)
            st.markdown("\n\n\n")
            draw_graph(formatted_predicted)

        with col3:
            with st.spinner('AI가 예측 결과를 분석 중입니다...'):
                prompt = generate_prompt(title, genres, director, ', '.join(cast), screenplay, original_work, runtime, rating,
                                        num_screens, nationality, series_value, predicted_int)
                ai_response = request_chat_completion(prompt)

                st.text_area(
                    label="AI가 예측 결과를 분석 중입니다...",
                    value=ai_response,
                    placeholder="AI가 예측 결과를 분석 중입니다...",
                    height=600,
                    label_visibility="collapsed"
                )
    else:
        _, col2, _ = st.columns([3, 8, 3])
        with col2:
            st.markdown("\n\n")
            st.markdown("\n\n")
            st.markdown(f"<h4 style='text-align: center;'>역대 개봉영화 대비 예상 흥행률</h4>", unsafe_allow_html=True)
            st.markdown("\n\n\n")
            draw_graph(formatted_predicted)
