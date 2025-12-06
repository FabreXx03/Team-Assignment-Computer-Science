SMI Stock & Portfolio Comparator - Project Group 05.10

Project Overview

The objective of our web application is to help investors in the Swiss Market Index (SMI) to make data-driven investment decisions. The application goes beyond simple price charts and analyses the risk-return dynamics of the SMI-assets and the correlation of the different assets. It furthermore integrates Machine Learing by volatility forecasting to help investors gain model-based insights about the future behavior of assets.

File Structure

SMI_Portfolio_Comparator_App.py: The main file containing all our code.

requirements.txt: A list of Python libraries required to run this application.

Doumentation_Project_Group_05.10.pdf: A 

README.txt: This instruction file.


How to Run the Application

Ensure you have Python installed.

Install the required dependencies by running the following command in your terminal:
pip install -r requirements.txt

Launch the application:
streamlit SMI_Portfolio_Comparator_App.py

Note on Data Loading

The application fetches live data from Yahoo Finance. Please ensure you have an active internet connection when running the app. The initial data load might take a few seconds depending on your connection speed.
