# Real Estate price prediction project in Madrid

![Real Estate](https://cdn.midjourney.com/4f7d8a55-1d31-4e80-b726-26050bde3bd8/0_1.png)

## 📜 Project Overview

This project aims to predict property prices in Madrid using a data science-based approach. Using a detailed dataset about properties, we implement processing, analysis, and modeling techniques to estimate housing prices. Additionally, visualizations are generated to explain the most influential factors affecting property prices. A Streamlit web application is included for interactive predictions.

The dataset includes information such as location, size, number of rooms, property type, among others. These variables are analyzed and transformed to train Machine Learning models capable of predicting housing prices with high accuracy.

### Specific Objectives

1. **Data Loading and Exploration:** Understand and analyze the dataset variables to identify patterns, trends, and outliers.
2. **Data Preprocessing:**
   - Clean and transform data.
   - Encode categorical variables.
   - Scale numerical variables.
   - Handle missing values and outliers.
3. **Model Building:**
   - Train predictive models such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting.
   - Evaluate models using metrics like RMSE and R².
4. **Visualization:** Generate charts explaining variable importance and model predictions.
5. **Optimization:** Tune hyperparameters to improve model performance.

## 💻 Project Structure

```
RealEstatePricePrediction
├── data/                               # Folder containing datasets
│   ├── raw/                            # Raw data files
│   ├── output/                         # Processed data outputs
├── models/                             # Saved trained models
├── notebook/                           # Jupyter Notebooks for exploratory data analysis and modeling
├── src/                                # Source code
│   ├── support_eda.py                  # Functions for exploratory data analysis
│   ├── support_preprocess.py           # Data cleaning and transformation functions
│   ├── support_regression.py           # Regression model functions
│   ├── support_st.py                   # Streamlit app support functions
├── .gitignore                          # Git ignore file
├── main.py                             # Main script for running the application
├── README.md                           # Project description and documentation
├── requirements.txt                    # Project dependencies
```

## 🔧 Installation and Requirements

This project was developed using Python 3.12. To set up the environment, follow these steps:

1. **Clone the repository:**

```bash
git clone https://github.com/SupernovaIa/Proyecto7-Housing-Market-Prediction-ML
```

2. **Navigate to the project directory:**

```bash
cd Proyecto7-Housing-Market-Prediction-ML
```

3. **Install the required dependencies:**

The following libraries are required for this project:

- [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- [NumPy](https://numpy.org/): For numerical data processing.
- [Matplotlib](https://matplotlib.org/): For data visualization.
- [Seaborn](https://seaborn.pydata.org/): For statistical visualization.
- [Streamlit](https://streamlit.io/): For building interactive web apps.
- [Category Encoders](https://contrib.scikit-learn.org/category_encoders/): For encoding categorical variables.
- [Scikit-learn](https://scikit-learn.org/): For Machine Learning algorithms.
- [Pickle](https://docs.python.org/3/library/pickle.html): For serializing and deserializing Python objects.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

4. Run the application: Once the environment is set up, you can  launch the Streamlit web app by running:

```bash
streamlit run main.py
```

5. Run the notebooks: Once the environment is set up, you can execute the notebooks in the specified order to perform data exploration, modeling, and visualization.

## 📊 Results and Conclusions

### Key Findings:

1. **Exploratory Analysis:**

   - There is a moderate positive correlation between rooms, bathrooms, and size, as larger homes typically have more rooms.

    - A weaker correlation exists between distance and size, indicating that more centrally located homes tend to be smaller.

    - Distance has a negative correlation with price, suggesting that homes farther from the city center are generally cheaper.

2. **Predictive Model Performance:**

   Below is a comparison of the model performances based on R² and RMSE:

   | Model               | R²   | RMSE  |
   |---------------------|-------|-------|
   | Linear Model        | 0.59  | 53.55 |
   | Decision Tree       | 0.71  | 44.90 |
   | Random Forest       | 0.81  | 36.31 |
   | Gradient Boosting   | 0.81  | 36.37 |

`Random Forest` and `Gradient Boosting` show the best performance.

3. **Additional Conclusions:**

We have tried to segment the dataset by focusing solely on houses in Madrid, which makes more sense for a prediction specific to this city. However, the dataset was reduced to 38%, resulting in a model with very poor performance.

### Conclusions:

The data seems to have been selected with a price cap of €750, which may have biased the model. Even so the predictive model built can estimate property prices in Madrid with moderate accuracy. 

However, a more detailed analysis is suggested for properties with extreme prices, and additional models should be explored to further improve performance.

## 🔄 Next Steps

1. **Incorporate Additional Data:**

- Expand the dataset to a less restrictive price range.

2. **Optimize Models:**

- Test advanced models like XGBoost and different preprocesing techniques to compare results.

3. **Web App:**

- Improve the web interface in Streamlit for better functionality and a property selector.

## 🤝 Contributions

Contributions to this project are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request when ready.

If you have suggestions or improvements, feel free to open an issue.

## ✍️ Author

Javier Carreira - Lead Developer - [GitHub](https://github.com/SupernovaIa)

We thank the real estate data platforms for providing the base information and Hack(io) for the opportunity to develop this project.

