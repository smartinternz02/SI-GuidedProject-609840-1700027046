
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Prediction")

with st.form("my_form"):
    balance=st.number_input(label='Balance',step=0.001,format="%.6f")
    balance_frequency=st.number_input(label='Balance Frequency',step=0.001,format="%.6f")
    purchases=st.number_input(label='Purchases',step=0.01,format="%.2f")
    oneoff_purchases=st.number_input(label='OneOff_Purchases',step=0.01,format="%.2f")
    installments_purchases=st.number_input(label='Installments Purchases',step=0.01,format="%.2f")
    cash_advance=st.number_input(label='Cash Advance',step=0.01,format="%.6f")
    purchases_frequency=st.number_input(label='Purchases Frequency',step=0.01,format="%.6f")
    oneoff_purchases_frequency=st.number_input(label='OneOff Purchases Frequency',step=0.1,format="%.6f")
    purchases_installment_frequency=st.number_input(label='Purchases Installments Freqency',step=0.1,format="%.6f")
    cash_advance_frequency=st.number_input(label='Cash Advance Frequency',step=0.1,format="%.6f")
    cash_advance_trx=st.number_input(label='Cash Advance Trx',step=1)
    purchases_trx=st.number_input(label='Purchases TRX',step=1)
    credit_limit=st.number_input(label='Credit Limit',step=0.1,format="%.1f")
    payments=st.number_input(label='Payments',step=0.01,format="%.6f")
    minimum_payments=st.number_input(label='Minimum Payments',step=0.01,format="%.6f")
    prc_full_payment=st.number_input(label='PRC Full Payment',step=0.01,format="%.6f")
    tenure=st.number_input(label='Tenure',step=1)

    data=[[balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,purchases_frequency,oneoff_purchases_frequency,purchases_installment_frequency,cash_advance_frequency,cash_advance_trx,purchases_trx,credit_limit,payments,minimum_payments,prc_full_payment,tenure]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust=loaded_model.predict(data)[0]
    print('Data Belongs to Cluster',clust)

    cluster_df1=df[df['Cluster']==clust]
    plt.rcParams["figure.figsize"] = (20,3)
    for c in cluster_df1.drop(['Cluster'],axis=1):
        fig, ax = plt.subplots()
        grid= sns.FacetGrid(cluster_df1, col='Cluster')
        grid= grid.map(plt.hist, c)
        plt.show()
        st.pyplot(figsize=(5, 5))

from flask import Flask, render_template, request
import numpy as np
import pickle

pickle.dump(, open('model.pkl', 'wb'))


app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl' , 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", method=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = float(request.form[Gender])
        Total_builirubin = float(request.form['Gender'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamzine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase  = int(request.form['Aspartate_Aminotransferase'])
        Total_protiens = float(request.form['Total_Protines'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globuin_Ratio = float(request.form('Albumin_and_Globuin_Ratio'))

        values = np.array([[Age,Gender,Total_builirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_protiens,Albumin,Albumin_and_Globuin_Ratio]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st

# # Load the data
# df = pd.read_csv("mcdonalds.csv")

# # Data preprocessing
# df["Gender"].replace("Female", 0, inplace=True)
# df["Gender"].replace("Male", 1, inplace=True)
# df["Like"].replace("I hate it!-5", -5, inplace=True)
# df["Like"].replace("I love it!+5", 5, inplace=True)

# # Data exploration
# st.header("Customer Segmentation Analysis")
# st.subheader("Data Exploration")

# # Gender distribution
# gender_counts = df["Gender"].value_counts()
# st.bar_chart(gender_counts)

# # Visit frequency distribution
# visit_freq_counts = df["VisitFrequency"].value_counts()
# st.bar_chart(visit_freq_counts)

# # Like distribution
# like_counts = df["Like"].value_counts()
# st.bar_chart(like_counts)

# # K-means clustering
# st.subheader("K-means Clustering")

# # Determining the optimal number of clusters
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1, 12)).fit(df)
# st.pyplot(visualizer)

# # Clustering using the optimal number of clusters
# kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0).fit(df)
# df["cluster_num"] = kmeans.labels_

# # Cluster characteristics
# st.subheader("Cluster Characteristics")

# # Cluster size
# cluster_sizes = Counter(kmeans.labels_)
# st.write(cluster_sizes)

# # Cluster visualization
# st.scatterplot(
#     x="pc1",
#     y="pc2",
#     c="cluster_num",
#     data=df,
#     palette="hls",
#     size="Age",
#     alpha=0.7,
# )

# # Audience in each cluster based on gender
# gender_crosstab = pd.crosstab(df["cluster_num"], df["Gender"])
# st.write(gender_crosstab)

# # Audience in each cluster based on age
# st.boxplot(x="cluster_num", y="Age", showmeans=True, data=df)

# # Mean visit frequency for each cluster
# visit_freq = df.groupby("cluster_num")["VisitFrequency"].mean()
# st.write(visit_freq)

# # Segment description
# segment = (
#     df.groupby("cluster_num")
#     .agg(Like=("Like", "mean"), Gender=("Gender", "mean"), VisitFrequency=("VisitFrequency", "mean"))
#     .reset_index()
# )
# st.write(segment)
