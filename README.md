# 🚀 Machine Learning: Student Performance Predictor 

Welcome to the **Machine Learning for Student Performance Predictor**. This is a machine learning algorithm for predicting student performance using the Linear Regression technique. The goal of this program is to forecast the final grades of students based on their academic performance and other related factors.

In this algorithm, we use the `student-mat.csv` dataset, which is part of the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance). The dataset contains information about student performance in mathematics. The features include attributes such as first-period grade, second-period grade, weekly study time, school type, family size, parent's occupation, and more.

**The project consists of two main components:**

- [`GUI.ipynb`](https://github.com/William-Laverty/ML-Student-Performance-Predictor/blob/main/gradePredictor%20-%20GUI.ipynb) This Jupyter Notebook contains a user-friendly graphical user interface (GUI) that allows users to input student data and obtain grade predictions through an interactive and intuitive interface. 

- [`TERMINAL.ipynb`](https://github.com/William-Laverty/ML-Student-Performance-Predictor/blob/main/gradePredictor%20-%20TERMINAL.ipynb) This Jupyter Notebook provides a command-line interface (CLI) version of the grade prediction tool, enabling users to input student data through the terminal and receive grade predictions.

The [terminal version](https://github.com/William-Laverty/ML-Student-Performance-Predictor/blob/main/gradePredictor%20-%20TERMINAL.ipynb) includes a more detailed explanation of the prediction process, accompanied by graphs and visualizations to provide users with a deeper understanding of the model's performance and results. Users can explore the data analysis and model evaluation in the terminal version to gain insights into the grade prediction tool's performance.

## 🎯 Steps Performed by the Code

The Student Grade Predictor is a tool that uses a Linear Regression model to predict the final grade of a student based on their first-period grade (G1), second-period grade (G2), and weekly study time. The model is trained on a dataset containing student information, and the user can input values for G1, G2, and study time through an interactive Graphical User Interface (GUI) to obtain the predicted final grade for a new student.

The predictor uses one-hot encoding for categorical variables and is trained on a dataset (assuming the dataset is in the same directory as the script) that is preprocessed to handle missing values or categorical variables.

1. **Data Loading:** The code reads the `"student-mat.csv"` file, which contains the student performance data, using the pandas library. The data is loaded into a DataFrame for further processing.

2. **Data Preprocessing:** The dataset may have missing values or categorical variables that need handling. The code preprocesses the data, converting categorical variables into numerical form using one-hot encoding. This transformation is necessary because most machine learning algorithms, including Linear Regression, require numerical inputs.

3. **Data Splitting:** The data is split into training and testing sets using the `train_test_split()` function from sklearn. This ensures that the model is trained on a subset of the data and evaluated on unseen data to assess its generalization performance.

4. **Model Training:** The Linear Regression model from sklearn is created and trained on the training data using the `fit()` method. The model aims to learn the relationships between the features and the target variable (final grade).

5. **Model Evaluation:** After training, the model's performance is evaluated using the test data. Two common evaluation metrics used are Mean Squared Error (MSE) and R-squared (R2). MSE measures the average squared difference between the predicted and actual grades, while R2 indicates how well the model explains the variance in the target variable.

6. **Example Prediction with GUI:** The code features an interactive GUI that allows users to input the first-period grade, second-period grade, and weekly study time of a new student. The model will predict their final grade (G3) based on these inputs, providing a convenient and user-friendly way to utilize the predictor.

## 🔨 Install the required packages

These packages are essential for different aspects of the project, from data handling and machine learning to creating an interactive GUI within the Jupyter notebook environment.

- **Pandas**  # Data manipulation and analysis
- **Numpy**   # Fundamental package for numerical computations
- **Scikit-learn**  # Machine learning library
- **IPywidgets**    # Interactive widgets for Jupyter notebooks
- **Ttkthemes**     # Theming extension for Tkinter

To install the required packages, execute the following commands:

```bash
pip install pandas numpy scikit-learn ipywidgets ttkthemes numpy matplotlib.pyplot
```

## 🧑‍💻 Usage

1. Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/cgs-ist/student-grade-predictor-William-Laverty.git
cd ML-Student-Performance-Predictor
```

2. Ensure you have the required packages installed (see the Installation section).

3. Run the Jupyter notebook `"Student_Performance_Predictor.ipynb"` and interact with the GUI to predict student performance.

### 📝 License

The code and documentation in this repository are licensed under the MIT License. You can find the full license text in the [LICENSE](LICENSE) file.

### 📚 References

Cortez, Paulo. (2014). Student Performance. UCI Machine Learning Repository. [Link](https://doi.org/10.24432/C5TG7T)
