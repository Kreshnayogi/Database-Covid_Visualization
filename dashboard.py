import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import geopandas as gpd
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Load Data
data = pd.read_csv("full_data.csv") 
data.sort_values(by="date", inplace=True)
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date'])

summarized_data = pd.read_csv('owid-covid-latest.csv')
summarized_data['last_updated_date'] = pd.to_datetime(summarized_data['last_updated_date'])

# Title
st.title("COVID-19 Record's Dashboard")
st.write("Data Provided by John Hopkins University and WHO")

unique_countries = data['location'].unique()
# Filtering Components
min_date = data["date"].min()
max_date = data["date"].max()
with st.sidebar:
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    country = st.multiselect(
        'Country/Region',
        unique_countries,
        default = 'Indonesia',
        placeholder='Choose a Country/Region'
    )

main_df = data[(data["date"] >= str(start_date)) & 
                (data["date"] <= str(end_date))&
                (data['location'].isin(country))]

st.subheader(f"Time Series Analysis Total Cases and Deaths by COVID-19 in {country}")
col1, col2 = st.columns(2)
with col1:
    #Total Cases
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=main_df, x='date', y='total_cases', hue='location', ax = ax)

        # Calculate three dates in the middle
    middle_dates = [start_date + timedelta(days=(end_date - start_date).days // 4 * i) for i in range(1, 4)]

        # Combine all the dates
    xticks = [start_date, *middle_dates, end_date]

        # Set x-axis ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in xticks])
    ax.set_xlabel("Date")

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylabel("Total Cases")

    ax.set_title("Total Cases", loc="center")
    st.pyplot(fig)


# --------------------------------------------------
with col2:
    #Total Deaths
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=main_df, x='date', y='total_deaths', hue='location', ax = ax)

    # Calculate three dates in the middle
    middle_dates = [start_date + timedelta(days=(end_date - start_date).days // 4 * i) for i in range(1, 4)]

        # Combine all the dates
    xticks = [start_date, *middle_dates, end_date]

    # Set x-axis ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in xticks])
    ax.set_xlabel("Date")

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')
    ax.set_ylabel("Total Deaths")

    ax.set_title("Total Deaths", loc="center")
    st.pyplot(fig)
st.subheader(f"Time Series Analysis New Cases of COVID-19 in {country}")
#-------------New Cases-------------
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=main_df, x='date', y='new_cases', hue='location', ax = ax)

# Calculate three dates in the middle
middle_dates = [start_date + timedelta(days=(end_date - start_date).days // 4 * i) for i in range(1, 4)]

# Combine all the dates
xticks = [start_date, *middle_dates, end_date]

# Set x-axis ticks
ax.set_xticks(xticks)
ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in xticks])
ax.set_xlabel("Date")

# Prevent y-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='y', style='plain')
ax.set_ylabel("New Cases")

ax.set_title("New Cases", loc="center")
st.pyplot(fig)
# -----------------------------------
st.subheader(f"Time Series Analysis New Deaths by COVID-19 in {country}")
#-------------New Deaths-------------
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=main_df, x='date', y='new_deaths', hue='location', ax = ax)

# Calculate three dates in the middle
middle_dates = [start_date + timedelta(days=(end_date - start_date).days // 4 * i) for i in range(1, 4)]

# Combine all the dates
xticks = [start_date, *middle_dates, end_date]

# Set x-axis ticks
ax.set_xticks(xticks)
ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in xticks])
ax.set_xlabel("Date")

# Prevent y-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='y', style='plain')
ax.set_ylabel("New Deaths")

ax.set_title("New Deaths", loc="center")
st.pyplot(fig)
# --------------------------------------------------
st.subheader(f"Comparing Life Expectancy and Human Development Index to Deaths by COVID-19")
col1, col2 = st.columns(2)
with col1:
    a = summarized_data[summarized_data['location'] != 'World']
    #Total deaths with Life Expectancy
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=a, y='total_deaths', x='life_expectancy', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')

    ax.set_title("Life Expectancy Compared to Total Deaths in Each Country (Until 2023)", loc="center")
    st.pyplot(fig)

with col2:
    a = summarized_data[summarized_data['location'] != 'World']
    #Total deaths with with HDI
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=a, y='total_deaths', x='human_development_index', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')

    ax.set_title("Human Development Index Compared to Total Deaths in each country (Until 2023)", loc="center")
    st.pyplot(fig)
st.subheader(f"Analysis of Life Expectancy Correlation with Human Development Index")
# ------------ HDI VS LIFE EXPECTANCY ------------ 
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=summarized_data, x='life_expectancy', y='human_development_index', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='y', style='plain')

ax.set_title("Human Development Index Compared to Life Expectancy in Each Country (Until 2023)", loc="center")
st.pyplot(fig)
# ------------ HDI VS LIFE EXPECTANCY ------------ 
# ------------ WORLD MAP ------------    
col1, col2 = st.columns(2)    
with col1:
    world['name'] = world['name'].str.strip()
    summarized_data['location'] = summarized_data['location'].str.strip()
    merged_data_hdi = world.merge(summarized_data, how='left', left_on='name', right_on='location')
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title('Human Development Index Heatmap')
    # Plot the map
    merged_data_hdi.plot(column='human_development_index', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, legend_kwds={"shrink": 0.4, "orientation": "horizontal"})
    # Adjust the size of the colorbar using cax parameter
    st.pyplot(fig)
    # ------------ WORLD MAP ------------
    # ------------ WORLD MAP ------------    
with col2:
    world['name'] = world['name'].str.strip()
    summarized_data['location'] = summarized_data['location'].str.strip()
    merged_data_hdi = world.merge(summarized_data, how='left', left_on='name', right_on='location')
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title('Life Expectancy Index Heatmap')
    # Plot the map
    merged_data_hdi.plot(column='life_expectancy', linewidth=0.8, ax=ax, edgecolor='0.8', cmap='OrRd', legend=True, legend_kwds={"shrink": 0.4, "orientation": "horizontal"})
    # Adjust the size of the colorbar using cax parameter
    st.pyplot(fig)
# ------------ WORLD MAP ------------
st.subheader(f"Analysis of Smokers in {country}")
# Set the maximum number of columns
max_columns = 3

# Calculate the number of rows based on the maximum number of columns
num_rows = -(-len(country) // max_columns)  # Ceiling division to calculate the number of rows

# Create dynamic columns based on the maximum number of columns
columns = st.columns(len(country))

Gender = st.selectbox(
    'Select Smokers by Gender',
    ('M','F'))
# Loop through the items and distribute them across columns and rows
for i, item in enumerate(country):
    col_index = i % max_columns
    row_index = i // max_columns
    selected_country_smoker = summarized_data[summarized_data['location'] == country[i]]

    if Gender == 'M':
        smoker_column = 'male_smokers'
    else:
        smoker_column = 'female_smokers'

    if selected_country_smoker[smoker_column].isnull().values.any():
        # Create a subplot in the current column
        with columns[col_index]:
            st.write(f"{Gender} Smoker percentage of {country[i]} is not available")
    else:
        with columns[col_index]:
            fig,ax = plt.subplots(figsize=(2,2))
            # Extract the 1D array (flatten) from the DataFrame
            smoker_data = selected_country_smoker[smoker_column].values.flatten()/100
            values = [smoker_data[0], 1 - smoker_data[0]]
            # Create a circle at the center of the plot

            # Give color names
            ax.pie(x=values, labels=['Smokers', 'Non-Smokers'], autopct='%1.0f%%')
            ax.set_title(f'Smoker Proportion in {country[i]}, ({Gender})')

            # Show the graph
            st.pyplot(fig)
st.subheader("Analysis of GDP per Capita and Extreme Poverty with COVID-Related Features")
col1, col2 = st.columns(2)    
with col1:
    a = summarized_data[summarized_data['location'] != 'World']
    #Total deaths with Life Expectancy
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=a, y='total_deaths', x='extreme_poverty', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')

    ax.set_title("Extreme Poverty Distribution Against Deaths by Covid (Until 2023)", loc="center")
    st.pyplot(fig)

with col2:
    a = summarized_data[summarized_data['location'] != 'World']
    #Total deaths with Life Expectancy
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=a, y='total_cases', x='gdp_per_capita', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
    ax.ticklabel_format(axis='y', style='plain')

    ax.set_title("GDP per Capita Distribution Against Deaths by Covid (Until 2023)", loc="center")
    st.pyplot(fig)

# ------------ GDP VS POVERTY ------------ 
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=summarized_data, x='gdp_per_capita', y='extreme_poverty', ax = ax)

    # Prevent x-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='x', style='plain')

    # Prevent y-axis labels from being displayed in scientific notation
ax.ticklabel_format(axis='y', style='plain')

ax.set_title("GDP per Capita Distribution Against Extreme Poverty (Until 2023)", loc="center")
st.pyplot(fig)
# ------------ GDP VS POVERTY ------------ 

# Footer
st.subheader("Work Done By:")
st.text("Kreshnayogi Dava Berliansyach\nLouis Widi Anandaputra\nTeuku Aldi Fadhlur Rahman\n\nAs a means of completing Database Course Final Task - 2023")
