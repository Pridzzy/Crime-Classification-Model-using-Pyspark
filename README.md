# Crime Classification Model using PySpark

This project implements an automated crime classification system using PySpark's MLlib. It leverages machine learning techniques and natural language processing (NLP) to categorize crime descriptions into 39 predefined categories, enabling more efficient resource allocation and incident response.

## Problem Statement

Urban areas face challenges in managing and responding to diverse criminal activities. Manual categorization of crime descriptions is time-consuming for law enforcement agencies. This project aims to automate the classification process, enhancing the speed and accuracy of crime data analysis.

## Features and Benefits

- **Efficiency Enhancement**: Automates the crime classification process, speeding up investigations and enabling targeted resource allocation.
- **Scalable Data Processing**: Utilizes PySpark for handling large datasets effectively.
- **Improved Accuracy**: Employs advanced machine learning techniques to minimize human error in crime categorization.
- **Standardized Approach**: Reduces dependency on subjective interpretation through systematic methods.

## Dataset

The dataset, sourced from Kaggle, consists of San Francisco crime data with the following features:
- **Dates**: Date of the crime.
- **Category**: Type of crime (e.g., Robbery, Assault).
- **Descript**: Detailed description of the crime.
- **DayOfWeek**: Day of the week of the crime.
- **PdDistrict**: Police district of the crime.
- **Resolution**: Action taken for the crime.
- **Address**: Location of the crime.
- **#X, #Y**: Coordinates of the crime location.

## Solution Design

### Modules

1. **Data Extraction**:
   - Load dataset into PySpark DataFrame.
   - Analyze the dataset for insights.
   
2. **Partitioning**:
   - Split data into training and test sets.

3. **Feature Extraction**:
   - **Tokenization**: Implemented using RegexTokenizer.
   - **Stopword Removal**: Using StopWordsRemover.
   - **Vectorization**: Features extracted using CountVectorizer, TF-IDF, and Word2Vec.

4. **Model Training and Evaluation**:
   - Train baseline logistic regression model.
   - Construct a PySpark pipeline for feature extraction and model training.
   - Evaluate model performance using metrics.

5. **Deployment**:
   - Deploy trained model for real-time crime classification.

## Technologies Used

- PySpark
- NumPy
- Pandas

## Inference and Future Work

### Inference
- Successful implementation of an automated crime classification system.
- Enhanced efficiency in resource allocation and incident response.

### Future Extensions
- Continuous model improvement with additional labeled data.
- Integration of real-time data sources for dynamic crime classification.
- Exploration of advanced NLP and machine learning techniques for better accuracy.
---

Feel free to extend or modify this document as per your project's additional features or findings!

