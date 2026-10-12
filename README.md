# -Customer-Segmentation-Using-K-Means-Clustering
An end-to-end unsupervised machine learning project that segments mall customers based on annual income and spending behavior. Includes exploratory data analysis, feature scaling, K-Means clustering, cluster evaluation, business recommendations, and an interactive Streamlit dashboard
# Customer Segmentation Using K-Means Clustering

An end-to-end unsupervised machine learning project that segments mall customers based on annual income and spending behavior. Includes exploratory data analysis, feature scaling, K-Means clustering, cluster evaluation, business recommendations, and an interactive Streamlit dashboard.

## Project Overview

Businesses need to understand different customer groups to develop relevant marketing strategies and improve customer engagement.

This project uses **K-Means clustering** to group customers according to their annual income and spending score. The resulting segments help illustrate how customer analytics can support targeted marketing decisions.

### Objectives

- Explore customer demographics and spending patterns.
- Clean and prepare data for machine learning.
- Identify customer groups using unsupervised learning.
- Select the number of clusters using the Elbow Method.
- Evaluate cluster separation using the Silhouette Score.
- Interpret the clusters and recommend potential marketing strategies.
- Present results through an interactive dashboard.

## Dataset

**Dataset:** Customer Segmentation Data  
**Source:** Kaggle  
**Dataset link:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

The dataset contains 200 customer records and five original columns:

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age in years |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Mall-assigned spending score |

The clustering model uses annual income and spending score. CustomerID is excluded because it is an identifier rather than a behavioral feature.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- K-Means clustering
- StandardScaler
- Streamlit

## Project Workflow

1. **Data loading and inspection:** Examine dataset dimensions, columns, data types, and missing values.
2. **Data cleaning:** Check missing values, duplicate records, unique customer IDs, and invalid values.
3. **Exploratory Data Analysis:** Analyze customer demographics, income distribution, spending scores, and feature relationships.
4. **Feature selection:** Select annual income and spending score for the initial clustering model.
5. **Feature scaling:** Apply StandardScaler to put both features on comparable scales.
6. **Choosing K:** Use the Elbow Method to examine the trade-off between the number of clusters and inertia.
7. **Model training:** Train K-Means with five clusters using `random_state=42` and `n_init=10`.
8. **Model evaluation:** Calculate the Silhouette Score to assess cluster separation.
9. **Cluster profiling:** Compare customer counts, average age, average income, and average spending score.
10. **Business interpretation:** Translate the resulting segments into potential marketing actions.
11. **Dashboard development:** Use Streamlit to explore segments and filter customer records interactively.

## Model Configuration and Evaluation

| Parameter | Value |
|---|---|
| Algorithm | K-Means |
| Number of clusters | 5 |
| Scaling method | StandardScaler |
| Random state | 42 |
| Initializations (`n_init`) | 10 |
| Silhouette Score | 0.5547 |

The Elbow Method was used to select five clusters as an initial modeling choice. The resulting Silhouette Score of **0.5547** indicates reasonably defined cluster separation, although some overlap may remain.

The silhouette score is not classification accuracy. K-Means is an unsupervised algorithm, and the dataset does not provide predefined customer-segment labels.

## Customer Segmentation Results

The following results were obtained from the trained model.

| Cluster | Customers | Average Age | Average Income ($k) | Average Spending Score |
|---|---:|---:|---:|---:|
| 0 | 81 | 42.72 | 55.30 | 49.52 |
| 1 | 39 | 32.69 | 86.54 | 82.13 |
| 2 | 22 | 25.27 | 25.73 | 79.36 |
| 3 | 35 | 41.11 | 88.20 | 17.11 |
| 4 | 23 | 45.22 | 26.30 | 20.91 |

### Business Insights and Recommendations

**Cluster 0 — Average customers**

These customers have moderate average income and spending scores. Consider loyalty programs, relevant product recommendations, and campaigns that encourage repeat purchases.

**Cluster 1 — High-income, high-spending customers**

These customers have high average income and the highest average spending score. Consider premium product recommendations, exclusive benefits, and loyalty rewards.

**Cluster 2 — Lower-income, high-spending customers**

These customers have lower average income but high spending scores and a younger average age. Consider personalized recommendations and promotions suited to their preferences.

**Cluster 3 — High-income, low-spending customers**

These customers have the highest average income but the lowest average spending score. Test targeted campaigns to investigate whether product relevance, shopping preferences, or other factors affect purchasing behavior.

**Cluster 4 — Lower-income, low-spending customers**

These customers have lower average income and low spending scores. Consider affordable products, value-oriented offers, and relevant recommendations.

These segment names are descriptive interpretations of the model output, not verified customer preferences or evidence of actual campaign performance.

## Interactive Dashboard

The Streamlit dashboard is designed to provide:

- KPI cards for customer count, average income, and average spending score.
- An interactive scatter plot of annual income versus spending score.
- Filters for customer segment, gender, age, and income.
- A table of customer segment profiles.
- A customer explorer and downloadable filtered records.
- Potential marketing recommendations for each segment.

## Repository Structure

```text
mall-customer-segmentation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── customer_segmentation.ipynb
├── images/
│   ├── customer_segments.png
│   ├── elbow_method.png
│   └── dashboard.png
└── outputs/
    └── cluster_summary.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mall-customer-segmentation.git
cd mall-customer-segmentation
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the CSV from Kaggle:

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Place the extracted `Mall_Customers.csv` file inside the `data/` directory.

Make sure `app.py` loads the file from that location:

```python
pd.read_csv("data/Mall_Customers.csv")
```

### 5. Run the Jupyter Notebook

Open `notebooks/customer_segmentation.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

### 6. Launch the dashboard

From the repository root, run:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Limitations and Future Improvements

- The dataset is small and represents a particular retail context.
- Spending Score is provided by the dataset; it is not a direct measurement of customer lifetime value.
- The current model uses two features, so other customer characteristics are not incorporated into clustering.
- K-Means is sensitive to feature scaling, initialization, and the chosen number of clusters.
- Silhouette Score alone does not establish that segments will improve business outcomes.
- Marketing recommendations should be tested against real campaign results before implementation.

Potential future improvements include comparing different feature combinations, evaluating alternative clustering algorithms, adding PCA visualizations, persisting a trained model, and deploying the dashboard online.

## Key Learnings

- Data preprocessing and exploratory data analysis.
- Feature scaling and distance-based clustering.
- Choosing cluster counts with the Elbow Method.
- Evaluating clusters using Silhouette Score.
- Interpreting unsupervised learning results.
- Translating data patterns into business recommendations.
- Building an interactive analytics dashboard with Streamlit.

*This project is intended for educational and portfolio purposes. Cluster assignments and business recommendations are exploratory and should be validated before use in real marketing decisions.*
