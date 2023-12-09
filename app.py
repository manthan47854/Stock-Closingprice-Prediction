import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import WeekdayLocator
from keras.models import load_model
import streamlit as st

eq_details = pd.read_csv("EQUITY_L.csv")
stock_symbols = eq_details.SYMBOL.to_list()

# Function to read the CSV file
def read_csv_file(file_name):
    try:
        data = pd.read_csv(file_name)
        return data
    except FileNotFoundError:
        print("Please check the name and try again.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to get business days
def get_business_days(start_date, end_date):
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    return business_days

if __name__ == "__main__":
    st.title("Stock Trend Prediction")

    # Search bar for stock symbol with suggestions
    user_input = st.selectbox('Enter Stock Symbol', stock_symbols)

    if user_input in stock_symbols:  # Check if the user-selected symbol is in the list of stock symbols
        selected_stock_symbol = user_input
        file_path = f"Data/{selected_stock_symbol}.csv"
        df = read_csv_file(file_path)

        if df is not None:
            # Convert 'Date' column to datetime object
            df['Date'] = pd.to_datetime(df['Date'])

            st.subheader('Closing Price vs Time Chart')
            
            # Create a Matplotlib figure and plot the 'Close' column
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df['Date'], df['Close'], label='Closing Price')

            # Set x-axis limits to cover the entire date range
            ax1.set_xlim(df['Date'].min(), df['Date'].max())

            # Format x-axis ticks as years
            business_days = get_business_days(df['Date'].min(), df['Date'].max())

            # Create a date locator to set the x-axis ticks as years
            years_locator = mdates.YearLocator()
            ax1.xaxis.set_major_locator(years_locator)

            # Set the x-axis tick labels to display the respective years
            years_format = mdates.DateFormatter('%Y')
            ax1.xaxis.set_major_formatter(years_format)

            plt.title(f'{user_input} Stock Close Price')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            plt.tight_layout()  # Ensure the labels fit properly
            st.pyplot(fig1)
            
            st.subheader('100-days Moving Average')
            
            # Calculate the 100-days Moving Average
            ma100 = df['Close'].rolling(100).mean()

            # Create a Matplotlib figure and plot the 100-days Moving Average with the 'Close' data
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
            ax2.plot(df['Date'], ma100, label='100-days Moving Average', color='red')

            # Set x-axis limits to cover the entire date range
            ax2.set_xlim(df['Date'].min(), df['Date'].max())

            # Format x-axis ticks as years
            business_days = get_business_days(df['Date'].min(), df['Date'].max())

            # Create a date locator to set the x-axis ticks as years
            years_locator = mdates.YearLocator()
            ax2.xaxis.set_major_locator(years_locator)

            # Set the x-axis tick labels to display the respective years
            years_format = mdates.DateFormatter('%Y')
            ax2.xaxis.set_major_formatter(years_format)

            plt.title(f'{user_input} 100-days Moving Average')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            plt.legend()  # Show legend with labels
            plt.tight_layout()  # Ensure the labels fit properly
            st.pyplot(fig2)
            
            st.subheader('200-days Moving Average')
            
            # Calculate the 200-days Moving Average
            ma200 = df['Close'].rolling(200).mean()

            # Create a Matplotlib figure and plot the 200-days Moving Average with the 'Close' data
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
            ax3.plot(df['Date'], ma100, label='100-days Moving Average', color='red')
            ax3.plot(df['Date'], ma200, label='200-days Moving Average', color='green')

            # Set x-axis limits to cover the entire date range
            ax3.set_xlim(df['Date'].min(), df['Date'].max())

            # Format x-axis ticks as years
            business_days = get_business_days(df['Date'].min(), df['Date'].max())

            # Create a date locator to set the x-axis ticks as years
            years_locator = mdates.YearLocator()
            ax3.xaxis.set_major_locator(years_locator)

            # Set the x-axis tick labels to display the respective years
            years_format = mdates.DateFormatter('%Y')
            ax3.xaxis.set_major_formatter(years_format)

            plt.title(f'{user_input} 200-days Moving Average')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            plt.legend()  # Show legend with labels
            plt.tight_layout()  # Ensure the labels fit properly
            st.pyplot(fig3)



        #data training and testing
            data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
            data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) 
            
        
        #Min-Max Scaling
            from sklearn.preprocessing import MinMaxScaler
            scaler= MinMaxScaler(feature_range=(0,1))
            
            data_training_array= scaler.fit_transform(data_training)
        
        
        #Loading trained model
            model = load_model('new_keras_model.h5')    
            
        #Testing   
            past_100_days=data_training.tail(100)
            final_df=pd.concat([past_100_days,data_testing],ignore_index=True)
            input_data=scaler.fit_transform(final_df)


            x_test=[]
            y_test=[]
            for i in range(100,input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_pridicted=model.predict(x_test)
            scaler=scaler.scale_ 
            scaler_factor=1/scaler[0]
            y_pridicted=y_pridicted*scaler_factor
            y_test=y_test*scaler_factor
        
        #Final Graph with Prediction
            st.subheader('Predictions Vs Original')
            fig2= plt.figure(figsize=(12,6))
            plt.plot(y_test,'blue',label='Original Price')
            plt.plot(y_pridicted,'red',label='Pridicted Price')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)

        else:
            st.write("Unable to proceed further due to errors in reading the CSV file.")
    else:
        st.write("Please enter a valid Stock Symbol.")
