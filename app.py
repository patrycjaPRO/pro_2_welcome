import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

from dotenv import dotenv_values
from ydata_profiling import ProfileReport
from openai import OpenAI
from datetime import datetime

#env = dotenv_values(".env")

# Wczytaj konfigurację
config = dotenv_values(".env")
OPENAI_API_KEY = config.get("OPENAI_API_KEY")
QDRANT_URL = config.get("QDRANT_URL")
QDRANT_API_KEY = config.get("QDRANT_API_KEY")

CSV_PATH = "pp_welcome.csv"



openai_client = OpenAI(api_key=OPENAI_API_KEY)



def generate_data_profiling_report():
    """Tworzy raport profilowania danych z CSV."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        profile = ProfileReport(df, title="Raport Profilowania pp_welcome.csv")
        profile.to_file("raport_pp_welcome.html")
        print("Raport zapisany jako raport_pp_welcome.html.")
    else:
        print(f"Plik {CSV_PATH} nie istnieje.")

# Stałe i ścieżki plików
MODEL_NAME = 'pp_welcome'
DATA = 'pp_welcome.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'pp_welcome.json'
QDRANT_COLLECTION_NAME = "pp_welcome"

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def get_all_participants():
    df = pd.read_csv(DATA, sep=',')
    model = get_model()
    df_with_clusters = predict_model(model, data=df)
    return df_with_clusters







# Wczytanie danych i modeli
model = get_model()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
df = get_all_participants()

# Sidebar Navigation
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Wybierz sekcję", ["Ankieta", "Wizualizacje", "Raport"])

# Strona "Ankieta"
if page == "Ankieta":
    st.title("Dashboard Powitalny")
    st.write("#### Brawo! Właśnie rozpoczynasz swoją przygodę z Data Science i AI!")

    image_path = "grafika2.png"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.write("""
    Razem z Tobą tę wyjątkową przygodę zaczyna wielu innych Podróżników.
    Wszystkie pytania są opcjonalne. Odpowiedz na pytania poniżej i zobacz, z kim spędzisz najbliższe ekscytujące tygodnie!
    """)

    st.subheader("Wypełnij ankietę")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Wiek", df['age'].dropna().unique())
        edu_level = st.selectbox("Wykształcenie", df['edu_level'].dropna().unique())
        gender = st.selectbox("Płeć", df['gender'].dropna().unique())

    with col2:
        experience = st.selectbox("Doświadczenie", df['experience'].dropna().unique())
        industry = st.selectbox("Branża", df['industry'].dropna().unique())
        motivation = st.selectbox("Motywacja", df['motivation'].dropna().unique())

    with col3:
        fav_place = st.selectbox("Ulubione miejsce", df['fav_place'].dropna().unique())
        fav_animals = st.selectbox("Ulubione zwierzęta", df['fav_animals'].dropna().unique())

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
        'experience': experience,
        'industry': industry,
        'motivation': motivation
    }])

    if st.button("Zapisz moje dane"):
        df_existing = pd.read_csv(DATA)
        df_updated = pd.concat([df_existing, person_df], ignore_index=True)
        df_updated.to_csv(DATA, index=False)
        st.session_state['person_df'] = person_df
        st.success("Mamy to! Welcome to the Jungle! Zajrzyj do sekcji Wizualizacje i Raport!")


# Strona "Wizualizacje"
if page == "Wizualizacje" and not df.empty:
    st.title("Kim jesteśmy?")
    st.write("#### Takich trzech, jak nas dwóch, to nie ma ani jednego ;) Czy znajdziesz wśród nas swoje alter ego?")
   
    # Wczytanie obrazu
    image_path = "grafika3.png"
    if os.path.exists(image_path):
        image2 = Image.open(image_path)
        st.image(image2, use_container_width=True)

    if 'person_df' not in st.session_state:
        st.warning("Najpierw uzupełnij i zapisz swoje dane w sekcji Ankieta!")
    else:
        person_df = st.session_state['person_df']
        cluster_pred = predict_model(model, data=person_df)
        cluster_id = cluster_pred["Cluster"].values[0]
        cluster_info = cluster_names_and_descriptions[str(cluster_id)]

        st.header(f"Twoja Grupa AI Kumpli to: {cluster_info['name']}")
        st.markdown(cluster_info['description'])

        cluster_members = df[df['Cluster'] == cluster_id]
        st.metric("Liczba uczestników klastra", len(cluster_members))

        tabs = st.tabs(["Zawodowo", "Ludzko", "Ty na mapie"])

        with tabs[0]:
            st.subheader("Jacy jesteście zawodowo?")
            c1, c2 = st.columns(2)
            with c1:
                fig_experience = px.histogram(cluster_members, x="experience", title="Rozkład doświadczenia")
                st.plotly_chart(fig_experience)
                fig_edu = px.histogram(cluster_members, x="edu_level", title="Rozkład wykształcenia")
                st.plotly_chart(fig_edu)

            with c2:
                fig_industry = px.histogram(cluster_members, x="industry", title="Rozkład branż")
                st.plotly_chart(fig_industry)
                fig_motivation = px.histogram(cluster_members, x="motivation", title="Rozkład motywacji")
                st.plotly_chart(fig_motivation)

        with tabs[1]:
            st.subheader("Jacy jesteście prywatnie?")
            c3, c4 = st.columns(2)
            with c3:
                fig_age = px.histogram(cluster_members, x="age", title="Rozkład wieku")
                st.plotly_chart(fig_age)
                fig_gender = px.histogram(cluster_members, x="gender", title="Rozkład płci")
                st.plotly_chart(fig_gender)
                
            with c4:
                fig_place = px.histogram(cluster_members, x="fav_place", title="Ulubione miejsca")
                st.plotly_chart(fig_place)
                fig_animals = px.histogram(cluster_members, x="fav_animals", title="Ulubione zwierzęta")
                st.plotly_chart(fig_animals)

        with tabs[2]:
        # wykres profesjonalny (już istniejący, poprawiłam nazwę zmiennej)
            sunburst_columns_prof = ["age", "edu_level", "experience", "industry"]
            fig_sunburst_prof = px.sunburst(
            cluster_members.dropna(subset=sunburst_columns_prof),
            path=sunburst_columns_prof,
            title="Twoja Grupa - profesjonalnie"
            )
            st.plotly_chart(fig_sunburst_prof)

        # wykres prywatny (tutaj też poprawka z usuwaniem NaN)
            sunburst_columns_priv = ["gender", "fav_animals", "fav_place"]
            fig_sunburst_priv = px.sunburst(
            cluster_members.dropna(subset=sunburst_columns_priv),
            path=sunburst_columns_priv,
            title="Twoja Grupa - prywatnie"
            )
            st.plotly_chart(fig_sunburst_priv)

# Raport Data Profiling
def generate_report(df):
    from ydata_profiling import ProfileReport
    return ProfileReport(df, minimal=True).to_html()

if page == "Raport":   
    st.title("Raport Data Profiling")
    st.write("Dla geeków i dla innych ciekawych świata liczb podrzucamy dwa raporty:")
    st.write("Pierwszy raport pokazuje dane wszystkich Odważnych Uczestników.")
    st.write("Drugi raport zawiera wyniki Twoich AI-Ziomków z klastra, do którego Ci najbliej wedle Jej Wysokości Sztucznej Inteligencji, lecz najpierw wypełnij dane w Ankiecie. Miłych wrażeń. Ciekawych odkryć! :)")

    image_path = "grafika4.png"
    if os.path.exists(image_path):
        image3 = Image.open(image_path)
        st.image(image3, use_container_width=True)

    report_all = generate_report(df)
    components.html(report_all, height=400, scrolling=True)

    if 'person_df' in st.session_state:
        person_df = st.session_state['person_df']
        cluster_pred = predict_model(model, data=person_df)
        cluster_id = cluster_pred["Cluster"].values[0]
        cluster_members = df[df['Cluster'] == cluster_id]

        report_cluster = generate_report(cluster_members)
        st.subheader("Raport dla Twojego klastra")
        components.html(report_cluster, height=400, scrolling=True)
    else:
        st.warning("Najpierw uzupełnij i zapisz swoje dane w sekcji Ankieta!")

