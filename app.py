import streamlit as st
import joblib
import numpy as np

# Load the trained LightGBM model
model = joblib.load('lgbm.keras')

# Custom CSS for the horizontal navigation bar
st.markdown("""
    <style>
        /* Horizontal Navbar */
        .navbar {
            background-color: #333;
            overflow: hidden;
            top: 0;
            width: 100%;
            position: fixed;
            z-index: 100;
        }

        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 20px;
            text-decoration: none;
            font-size: 17px;
        }

        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }

        /* Content below the navbar */
        .content {
            margin-top: 60px;
        }

        .sidebar .sidebar-content {
            padding-top: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
    <div class="navbar">
        <a href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#info">Info</a>
        <a href="#prediction">Prediction</a>
    </div>
""", unsafe_allow_html=True)

# Streamlit Sidebar for navigation
page = st.sidebar.radio("Choose a page", ["Home", "About", "Info", "Prediction"])

# Home Page
if page == "Home":
    st.markdown("<div class='content'></div>", unsafe_allow_html=True)
    st.title('FraudSense: Credit Card Fraud Detection using Machine Learning Techniques')
    st.write("""
        Welcome to the **Fraud Detection System**!
        
        This is a semi-advanced tool that uses machine learning to identify fraudulent transactions in real-time. 
        Fraudulent transactions are a significant issue in financial systems, and detecting them efficiently can save businesses from significant financial losses.

        **How It Works:**
        
        The system utilizes a **LightGBM model**, which is a powerful gradient boosting algorithm known for its speed and accuracy in handling large datasets. 
        It has been trained on a dataset of transactions, where each transaction is represented by 5 important features that capture the essence of the transaction. 
        The model predicts whether a transaction is **fraudulent (1)** or **non-fraudulent (0)** based on these features.

        **How to Use This Application:**
        
        1. **Enter Transaction Features**: 
            - You will be prompted to enter 5 specific numerical features for a given transaction. These features have been abstracted for privacy and security reasons to ensure that no sensitive information about the transaction is exposed.
            - Each feature represents an important aspect of the transaction, such as transaction amount, frequency of similar past transactions, etc.
            
        2. **Click on 'Predict'**: 
            - Once you input the 5 features, simply click on the **'Predict'** button to trigger the fraud detection model.
            - The system will process the input and give you an instant result showing whether the transaction is **fraudulent** or **safe**.

        3. **View the Result**: 
            - After prediction, the system will display whether the transaction is likely fraudulent or not based on the model's evaluation of the input data.

        **Important Note on Privacy:**
        - The names of the features and their detailed descriptions are deliberately abstracted. This is done to protect sensitive transactional data and to maintain privacy. 
        - Only the prediction results are shared, ensuring that no sensitive information is exposed in the process.
        
        **The Benefits of Using This System:**
        
        - **Accurate Predictions**: Leveraging advanced machine learning techniques, this system provides high accuracy in detecting fraudulent transactions.
        - **Real-time Detection**: It can be used in real-time for businesses to monitor transactions as they occur.
        - **Improved Security**: By automating fraud detection, it helps reduce the chances of fraud slipping through unnoticed.
        
        **Additional Features**:
        - The system provides an easy-to-use interface where users can interact with the model, get predictions, and learn more about the underlying machine learning model (LightGBM).
        
        We hope you find this tool helpful for detecting fraudulent transactions and enhancing security.
        
        **To get started, simply enter the transaction details in the form and click 'Predict'!**
    """)

elif page == "About":
    st.markdown("<div class='content'></div>", unsafe_allow_html=True)
    st.title('About This Fraud Detection System')
    st.write("""
        ### Fraud Detection System Using Machine Learning

        This **Fraud Detection System** is built using a **LightGBM** model, an advanced machine learning algorithm known for its efficiency and accuracy. The goal of this system is to help detect fraudulent transactions in real-time by predicting whether a given transaction is fraudulent or not based on five key features.

        **Model Details:**
        - The model has been trained on a dataset of financial transactions. These transactions are represented by several features that capture various aspects of the transaction, such as amounts, frequency, and other transaction details.
        - The system uses the **LightGBM algorithm**, a gradient boosting model that provides high accuracy, speed, and scalability in large datasets.
        - The model analyzes these 5 selected features and generates a prediction: **Fraudulent (1)** or **Non-Fraudulent (0)**.
        
        **Why is Fraud Detection Important?**
        
        Fraud detection is a critical aspect of financial systems. As digital transactions continue to increase globally, the risk of fraud also rises. Fraudulent transactions can result in significant financial losses, and early detection is key to preventing such issues. This system is designed to provide quick, reliable predictions that can help businesses identify potentially fraudulent activities before they lead to losses.

        **Key Features:**
        - **High Accuracy**: The LightGBM algorithm is particularly well-suited for large datasets and provides highly accurate predictions.
        - **Real-Time Detection**: This system can be used in real-time to detect fraudulent activities as transactions occur.
        - **Privacy and Security**: The system abstracts sensitive details of the features to ensure the privacy of the transaction data.

        ### Team Contributors:
        This project was developed as part of the **Project-Based Learning (PBL)** course using machine learning techniques. The following team members contributed to building this system:

        - **G Pragna (1DT23CD020)**: Data Preprocessing, Feature Engineering, and Model Training.
        - **Pallavi VS (1DT23CD032)**: Data Collection, Feature Selection, and Model Evaluation.
        - **Raaja Nithila Nethran (1DT23CD039)**: Model Tuning, Cross-validation, and Deployment.
        - **Varsha V (1DT23CD059)**: Interface Design, Application Structure, and Documentation.
        
        Each member played an important role in the development of this system, collaborating closely to ensure its success. Their contributions include data preparation, model training and evaluation, creating the user interface, and ensuring the application runs smoothly for real-time predictions.

        **Conclusion:**
        
        The **Fraud Detection System** is an advanced tool that can be used to safeguard businesses and individuals from fraudulent transactions. By leveraging machine learning, specifically LightGBM, we have developed a tool that provides fast, reliable, and secure predictions.
    """)


elif page == "Info":
    st.markdown("<div class='content'></div>", unsafe_allow_html=True)
    st.title('Information About Fraud Detection System and LightGBM Algorithm')

    st.write("""
        ## Model Information
        **Model:** LightGBM Classifier  
        **Features:** 5 selected features from the dataset  
        **Input:** Transaction data consisting of 5 features  
        **Output:** Prediction (1: Fraudulent, 0: Not Fraudulent)
        
        ### How it Works:
        1. You input the 5 features of a transaction.
        2. The model processes this data using the trained **LightGBM algorithm**.
        3. Based on the input, it predicts if the transaction is fraudulent or not.

        ## About the LightGBM Algorithm

        **LightGBM (Light Gradient Boosting Machine)** is a gradient boosting framework that uses tree-based learning algorithms. It is designed for efficiency and scalability, particularly when dealing with large datasets.

        ### How LightGBM Works:
        1. **Gradient Boosting**: LightGBM builds an ensemble of decision trees, where each tree corrects the errors made by the previous one.
        2. **Leaf-wise Growth**: Unlike other algorithms like XGBoost that grow trees level-wise, LightGBM grows trees leaf-wise. This often results in deeper trees, which help in better model performance.
        3. **Histogram-based Decision Trees**: LightGBM uses a histogram-based approach to speed up the process of training. It buckets continuous values into discrete bins, allowing for faster computations.
        4. **Categorical Feature Support**: LightGBM natively supports categorical features, which can be used directly in the model without one-hot encoding.

        ### Key Benefits of LightGBM:
        - **Speed**: LightGBM is faster than many other gradient boosting algorithms, especially with large datasets.
        - **Accuracy**: LightGBM can provide high accuracy with less tuning.
        - **Memory Efficiency**: Due to its histogram-based approach, it uses less memory compared to other boosting algorithms.

        ### Hyperparameters to Tune in LightGBM:
        - **num_leaves**: The number of leaves in a tree. A higher value can result in overfitting.
        - **learning_rate**: The step size at each iteration while moving toward a minimum of a loss function.
        - **max_depth**: The maximum depth of each tree. Helps prevent overfitting.

        ### LightGBM for Fraud Detection:
        - LightGBM is particularly useful in fraud detection due to its ability to efficiently handle imbalanced datasets (fraudulent transactions are often much fewer than legitimate ones).
        - It performs well with both **numerical** and **categorical** data, making it ideal for this use case where different features may have different types of data.
    """)

# Prediction Page
elif page == "Prediction":
    st.markdown("<div class='content'></div>", unsafe_allow_html=True)
    st.title('Transaction Prediction')
    st.write("""
        Enter the 5 features of the transaction to predict if it is fraudulent or not.
    """)

    # Create input fields for the 5 features
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)
    feature_3 = st.number_input("Feature 3", value=0.0)
    feature_4 = st.number_input("Feature 4", value=0.0)
    feature_5 = st.number_input("Feature 5", value=0.0)

    # Collect inputs into a list
    inputs = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

    # Prediction and output
    if st.button('Predict'):
        prediction = model.predict(inputs)
        if prediction == 1:
            st.error("⚠️ The transaction is predicted to be **FRAUDULENT**.")
        else:
            st.success("✅ The transaction is predicted to be **NOT FRAUDULENT**.")