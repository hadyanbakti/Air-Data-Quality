import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

@st.cache_data
def input_data():
    df_all_clean = pd.read_csv("dashboard/clean_merged_dataset.csv")
    df_Aotizhongxin = pd.read_csv("Air-quality-dataset/PRSA_Data_Aotizhongxin_20130301-20170228.csv")
    df_Changping = pd.read_csv("Air-quality-dataset/PRSA_Data_Changping_20130301-20170228.csv")
    df_Dingling = pd.read_csv("Air-quality-dataset/PRSA_Data_Dingling_20130301-20170228.csv")
    df_Dongsi = pd.read_csv("Air-quality-dataset/PRSA_Data_Dongsi_20130301-20170228.csv")
    df_Guanyuan = pd.read_csv("Air-quality-dataset/PRSA_Data_Guanyuan_20130301-20170228.csv")
    df_Gucheng = pd.read_csv("Air-quality-dataset/PRSA_Data_Gucheng_20130301-20170228.csv")
    df_Huairou = pd.read_csv("Air-quality-dataset/PRSA_Data_Huairou_20130301-20170228.csv")
    df_Nongzhanguan = pd.read_csv("Air-quality-dataset/PRSA_Data_Nongzhanguan_20130301-20170228.csv")
    df_Shunyi = pd.read_csv("Air-quality-dataset/PRSA_Data_Shunyi_20130301-20170228.csv")
    df_Tiantan = pd.read_csv("Air-quality-dataset/PRSA_Data_Tiantan_20130301-20170228.csv")
    df_Wanliu = pd.read_csv("Air-quality-dataset/PRSA_Data_Wanliu_20130301-20170228.csv")
    df_Wanshouxigong = pd.read_csv("Air-quality-dataset/PRSA_Data_Wanshouxigong_20130301-20170228.csv")



    return (
        df_all_clean,
        df_Aotizhongxin,
        df_Changping,
        df_Dingling,
        df_Dongsi,
        df_Guanyuan,
        df_Gucheng,
        df_Huairou,
        df_Nongzhanguan,
        df_Shunyi,
        df_Tiantan,
        df_Wanliu,
        df_Wanshouxigong,
    )


(
    df_all_clean,
    df_Aotizhongxin,
    df_Changping,
    df_Dingling,
    df_Dongsi,
    df_Guanyuan,
    df_Gucheng,
    df_Huairou,
    df_Nongzhanguan,
    df_Shunyi,
    df_Tiantan,
    df_Wanliu,
    df_Wanshouxigong,
) = input_data()

st.sidebar.title("Air Quality Index")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    [
        "Home",
        "Show Dataset",
        "Pertanyaan 1",
        "Pertanyaan 2",
        "Pertanyaan 3",
        "Kesimpulan",
    ],
)

wilayah_dict = {
    "Aotizhongxin": df_Aotizhongxin,
    "Changping": df_Changping,
    "Dingling": df_Dingling,
    "Dongsi": df_Dongsi,
    "Guanyuan": df_Guanyuan,
    "Gucheng": df_Gucheng,
    "Huairou": df_Huairou,
    "Nongzhanguan": df_Nongzhanguan,
    "Shunyi": df_Shunyi,
    "Tiantan": df_Tiantan,
    "Wanliu": df_Wanliu,
    "Wanshouxigong": df_Wanshouxigong,
}

def pertanyaan_1():
    st.title("Bagaimana korelasi antara suhu (TEMP) dengan tingkat polutan utama (PM2.5, PM10, SO2, NO2, CO, O3)?")
    
    # Menghitung korelasi
    def analyze_seasonal_correlations(df):
        df['season'] = pd.cut(df['month'], 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
        seasons = df['season'].unique()
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
        seasonal_correlations = {season: {} for season in seasons}
    
        for season in seasons:
            season_data = df[df['season'] == season]
            for pollutant in pollutants:
                correlation, _ = stats.pearsonr(season_data['TEMP'], season_data[pollutant])
                seasonal_correlations[season][pollutant] = correlation
    
    
        plt.figure(figsize=(15, 8))
        x = np.arange(len(pollutants))
        width = 0.2
    
        for i, season in enumerate(seasons):
            plt.bar(x + i*width, list(seasonal_correlations[season].values()), 
                width, label=season)
    
        plt.xlabel('Pollutan')
        plt.ylabel('korelasi dengan temperatur')
        plt.title('Korelasi musim terhadap temperatur dan polutan')
        plt.xticks(x + width*1.5, pollutants, rotation=45)
        plt.legend()
    analyze_seasonal_correlations(df_all_clean)
    st.pyplot(plt)
    
    # Insight
    st.subheader("Insight:")
    st.write("""
    1. O3 menunjukkan korelasi positif yang kuat dengan suhu, mengindikasikan peningkatan pembentukan ozon pada suhu yang lebih tinggi.
    2. CO dan NO2 memiliki korelasi negatif dengan suhu, yang mungkin disebabkan oleh:
       - Peningkatan penggunaan pemanas pada suhu rendah
       - Kondisi inversi suhu yang menjebak polutan pada suhu rendah
    3. PM2.5 dan PM10 menunjukkan korelasi negatif lemah dengan suhu, menandakan pengaruh minimal suhu terhadap partikel tersuspensi.
    """)

def pertanyaan_2():
    st.title("Apakah ada perbedaan signifikan dalam tingkat polutan antara hari kerja dan akhir pekan?")
    
    # Menambahkan kolom untuk hari dalam seminggu
    df_all_clean['is_weekend'] = df_all_clean['day'].isin([6, 7]).map({True: 'Akhir Pekan', False: 'Hari Kerja'})
    
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, pollutant in enumerate(pollutants):
        sns.boxplot(x='is_weekend', y=pollutant, data=df_all_clean, ax=axes[idx])
        axes[idx].set_title(f'{pollutant} - Hari Kerja vs Akhir Pekan')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistical analysis
    st.subheader("Analisis Statistik:")
    for pollutant in pollutants:
        weekday_mean = df_all_clean[df_all_clean['is_weekend'] == 'Hari Kerja'][pollutant].mean()
        weekend_mean = df_all_clean[df_all_clean['is_weekend'] == 'Akhir Pekan'][pollutant].mean()
        percent_diff = ((weekend_mean - weekday_mean) / weekday_mean) * 100
        
        st.write(f"{pollutant}:")
        st.write(f"- Rata-rata hari kerja: {weekday_mean:.2f}")
        st.write(f"- Rata-rata akhir pekan: {weekend_mean:.2f}")
        st.write(f"- Perubahan: {percent_diff:.1f}%")

def pertanyaan_3():
    st.title(
        "Bagaimana tren tahunan tingkat rata-rata CO di berbagai kota dari 2013 hingga 2017 dan apakah ada pola umum yang terlihat?"
    )

   
    yearly_co = df_all_clean.groupby(['year', 'station'])['CO'].mean().unstack()

   
   

    # Plotting
    plt.figure(figsize=(15, 10))
    yearly_co.plot(marker="o")
    plt.title("Rata-rata Tingkat CO per Tahun untuk Setiap Kota")
    plt.xlabel("Tahun")
    plt.ylabel("Rata-rata Tingkat CO")
    plt.xticks(yearly_co.index.astype(int), rotation=45)
    plt.legend(title="Kota", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()


    st.pyplot(plt)

    for city in yearly_co.columns:
        total_change = yearly_co[city].iloc[-1] - yearly_co[city].iloc[0]
        st.write(
            f"Perubahan keseluruhan tingkat CO untuk {city} dari {yearly_co.index[0]} ke {yearly_co.index[-1]}: {total_change:.2f}"
        )

    # Insight/Kesimpulan
    st.subheader("Insight:")
    st.markdown(
        """
    Dari tren yang terlihat untuk CO terdapat penurunan pada tahun 2013 hingga 2016 tetapi terdapat kenaikan drastis di semua wilayah dari tahun 20136 sampai 2017.
    Hal ini menunjukkan bahwa keberhasilan pada berbagai wilayah untuk mengurangi CO pada tahun 2013 hingga tahun 2016, tetapi karena kendaraan sangat merajalela, pada tahun 2016
    hingga 2017 semua wilayah mengalami kenaikan CO yang drastis. Kenaikan paling terasa di wilayah Wanliu. sedangkan Gucheng menjadi wilayah dengan jumlah CO tertinggi.
    """
    )

def conclusion():
    st.title("Kesimpulan Komprehensif Analisis Kualitas Udara")
    
    st.subheader("Temuan Utama:")
    st.markdown("""
    1. **Korelasi Suhu dan Polutan**
       - O3 meningkat pada suhu tinggi
       - CO dan NO2 lebih tinggi pada suhu rendah
       - PM2.5 dan PM10 kurang dipengaruhi suhu

    2. **Perbedaan Hari Kerja vs Akhir Pekan**
       - NO2 dan CO menunjukkan penurunan di akhir pekan
       - O3 cenderung lebih tinggi di akhir pekan
       - PM2.5 dan PM10 relatif konsisten

    3. **perubahan CO setiap tahun**
       - Pada setiap tahunnya, CO tidak stabil dalam perubahannya
       - Pada tahun 2016 CO di semua wilayah mengalami penurunan
       - pada tahun 2017, CO di semua wilayah meningkat drastis dan menjadi tertinggi
    """)
    
    st.subheader("Rekomendasi:")
    st.markdown("""
    1. **Manajemen Berdasarkan Waktu**
       - Implementasi kebijakan berbasis waktu (hari kerja vs akhir pekan)
       - Pertimbangkan faktor cuaca dalam perencanaan pengendalian polusi

    2. **Perencanaan Spasial**
       - perbanyak transportasi umum sehingga mengurangi jumlah polusi
       - Fokuskan upaya pengurangan emisi di area sumber utama
       - gunakan data perkembangan CO untuk mengurangi produksi CO

    3. **Monitoring dan Peringatan**
       - Kembangkan sistem peringatan dini berbasis prediksi cuaca
       - Tingkatkan monitoring di area dengan risiko tinggi
    """)

    st.subheader("Implikasi Penelitian Lebih Lanjut:")
    st.markdown("""
    1. Analisis temporal jangka panjang untuk identifikasi tren
    2. Studi korelasi dengan faktor sosio-ekonomi
    3. Pengembangan model prediktif berbasis machine learning
    """)

# Main execution
if menu == "Home":
    st.title("Dashboard For Air Quality Index")
    st.markdown("""
       Air Quality atau kualitas udara adalah ukuran kebersihan udara
       di lingkungan tersebut. Hal-hal yang mempengaruhi kualitas udara
       antara lain partikel debu, asap, gas beracun, dan zat kimia
       lainnya. Pada kali ini, saya mempelajari data Air Quality Index
       yang ada di Beijing, China untuk menganalisis kualitas udara 
       yang ada di wilayah tersebut.
    """)
    st.subheader("Deskripsi Data")
    st.write(df_all_clean.describe())
    st.subheader("korelasi polutant dan suhu")
    analyze_seasonal_correlations(df_all_clean)

elif menu == "Show Dataset":
    st.title("Air Quality Dataset berdasarkan Wilayah di Beijing, China.")
    selected_wilayah = st.sidebar.selectbox("Pilih Wilayah:", list(wilayah_dict.keys()))
    st.subheader(f"Wilayah: {selected_wilayah}")
    st.subheader("Deskripsi Data")
    st.write(wilayah_dict[selected_wilayah].describe())
    st.subheader(f"Head of {selected_wilayah}")
    st.dataframe(wilayah_dict[selected_wilayah].head())

elif menu == "Pertanyaan 1":
    pertanyaan_1()
elif menu == "Pertanyaan 2":
    pertanyaan_2()
elif menu == "Pertanyaan 3":
    pertanyaan_3()
elif menu == "Kesimpulan":
    conclusion()
