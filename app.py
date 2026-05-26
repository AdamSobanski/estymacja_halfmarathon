import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pycaret.regression import load_model, predict_model
import openai
load_dotenv() 
import datetime

from langfuse.decorators import observe
from langfuse.openai import OpenAI

MODEL_NAME = 'model_bieg'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

model = get_model()

st.header("Estymacja czasu na mecie🦾🦾🦾!!!")

user_input = st.text_area('Przedstaw się nam! Podaj płeć, wiek i czas na 5km:')

openai.api_key = os.getenv("OPENAI_API_KEY")

@observe()
def extract_data(user_input):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Wyłuskaj dane z tekstu i zwróć JSON:
            {"Płeć": "M" lub "K", "Wiek": liczba, "Czas 5 km": "MM:SS", "Imię": "imię"}
            Czas na 5km zawsze zwróć w formacie MM:SS, niezależnie od formatu wejściowego.
            Jeśli brakuje danych zwróć null dla danego pola."""},
            {"role": "user", "content": user_input}
],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    if ':' in time:
        parts = time.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(time)

def kat_wiek(gender, age):
    age = int(age)
    if age < 10: zakres = '10'
    elif age < 20: zakres = '20'
    elif age < 30: zakres = '30'
    elif age < 40: zakres = '40'
    elif age < 50: zakres = '50'
    elif age < 60: zakres = '60'
    elif age < 70: zakres = '70'
    elif age < 80: zakres = '80'
    elif age < 90: zakres = '90'
    else: zakres = '100'
    return f"{gender}{zakres}"

if st.button('Oblicz czas'):
    data = extract_data(user_input)

    # st.dataframe(pd.DataFrame([data]).drop(columns=['Imię']), hide_index=True, use_container_width=True)

    st.subheader(f"Witaj {data['Imię']}!")
    # st.dataframe(pd.DataFrame([data]).drop(columns=['Imię']).style.set_properties(**{'text-align': 'center'}),
    #              hide_index=True)
    
#     st.markdown(
#     pd.DataFrame([data]).drop(columns=['Imię']).to_html(index=False, justify='center'),
#     unsafe_allow_html=True
# )
    
    bledy = [k for k, v in data.items() if v is None]
    if bledy:
        st.error(f"Brakuje danych: {', '.join(bledy)}")
    else:
        kategoria = kat_wiek(data['Płeć'], data['Wiek'])
        
        person_df = pd.DataFrame([{
            '5 km Czas': convert_time_to_seconds("00:" + data['Czas 5 km']),
            'Płeć': data['Płeć'],
            'Kategoria wiekowa': kategoria,
        }])
        
        prediction = predict_model(model, data=person_df)['prediction_label'].values[0]
        czas = str(datetime.timedelta(seconds=int(prediction)))
        # st.success(f"Twój estymowany czas ukończenia Półmaratonu: {czas}")
        # st.markdown(f"<h2 style='color: green; text-align: center;'>🏅 Estymowany czas ukończenia Półmaratonu: {czas}</h2>", unsafe_allow_html=True)
        # st.markdown(f"<h2 style='color: green; text-align: center;'>Estymowany czas ukończenia Półmaratonu: {czas}<br>🏅</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: green; text-align: center;'>Estymowany czas ukończenia Półmaratonu:<br><span style='color: gold;'>{czas}</span><br>🏅</h2>", unsafe_allow_html=True)