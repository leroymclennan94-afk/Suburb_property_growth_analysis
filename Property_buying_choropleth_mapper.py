import numpy as np
import pandas as pd
import math
import folium
import json

# package versions = package_versions.txt

###############################################################
#################### Read in the data #########################
###############################################################

# House Price CSV File import details as below
h_file_path: str = "C:/Users/leroy/Downloads/Houses-by-suburb-2013-2023.xlsx - Table 1.csv"
h_column_prefix: str = 'H_'
h_col_ind_to_read = [0, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 15]  # Read these rows from house CSV
h_rows_to_skip: int = 4  # Skip the first 4 rows when reading from CSV

# Unit Price CSV File import details as below
u_file_path = "C:/Users/leroy/Downloads/units-by-suburb-2014-2024.xlsx - Table 1.csv"
u_column_prefix: str = 'U_'
u_col_ind_to_read = [0, 2, 3, 4, 5, 7, 9, 11, 12,
                     13, 14, 24]  # Read these rows from unit CSV
u_rows_to_skip: int = 2  # Skip the first 2 rows when reading from CSV
growth_col: int = 11  # The growth rate for each is stored in the 11th column
col_names = ['Suburb', '2014', '2015', '2016', '2017',
             '2018', '2019', '2020', '2021', '2022', '2023_Prop_Price', 'Annual_Growth_Rate']


def csv_importer(file_path: str, col_ind: list[int], rows_to_skip: int, col_names: list[str], column_name_prefix: str) -> pd.DataFrame:
    """
    This function takes a file path, list of column indices, number of rows to skip, and column names, 
    and returns a DataFrame containing the data from the CSV file with the desired columns and column names.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be read.
    col_ind : list
        A list of column indices to be read from the CSV file.
    rows_to_skip : int
        The number of rows to skip when reading from the CSV file.
    col_names : list
        A list of column names to be used when reading from the CSV file.

    Returns
    -------
    DataFrame
        A DataFrame containing the data from the CSV file.
    """
    # Set the prefixes for the column names - For house file the prefix is 'H_', for unit file the prefix is 'U_' - This is the 25th character in the file path string
    col_names_prefixed = [
        'Suburb']+[column_name_prefix + col for col in col_names[1:]]
    # Write the CSV to a dataframe
    df: pd.DataFrame = pd.read_csv(file_path, usecols=col_ind,
                                   names=col_names_prefixed, skiprows=rows_to_skip)
    return df


# Import the files as DataFrames
house_price_df = csv_importer(
    h_file_path, h_col_ind_to_read, h_rows_to_skip, col_names, h_column_prefix)
unit_price_df = csv_importer(
    u_file_path, u_col_ind_to_read, u_rows_to_skip, col_names, u_column_prefix)


def dataframe_tidyup(df: pd.DataFrame) -> pd.DataFrame:
    # Set the dtype for each column
    """
    Process the DataFrame by setting the dtype for each column, removing non-digit-characters, 
    converting to numeric, setting the type to float64, and getting rid of bracketed suburb names.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be processed.

    Returns
    -------
    DataFrame
        The processed DataFrame.
    """

    # Tidy up the property value data to allow future operations
    for col_name in df.columns[1:-1]:
        # Remove everything except numbers
        df[col_name] = df[col_name].replace(r"\D+", '', regex=True)
        # Convert to numeric, setting errors to NaN
        df[col_name] = df[col_name].apply(pd.to_numeric, errors='coerce')
        # Set the type to float64
        df[col_name] = df[col_name].astype(
            'float64')

    # Some suburbs are named as such "Suburb (Suburb)" - Remove the contents of brackets (and the brackets too) from the suburb string names
    for name in df['Suburb']:
        # Remove the contents of brackets (and the brackets too) from the suburb string names and remove trailing spaces
        df.iloc[df['Suburb'] == name, 0] = name.split('(', 1)[0].strip()

    return df


# Deal with outliers before tidying up the rest of the document
# Modify the kew north entry, this element contains 2 values merged in the same column
kew_north_val_str = str(unit_price_df.iloc[229, 3])
unit_price_df.iloc[229, 3], unit_price_df.iloc[229,
                                               4] = float(kew_north_val_str[:6]), float(kew_north_val_str[6:])
# Semi-duplicate entry that is actually the same suburb - Remove it
house_price_df = house_price_df[house_price_df['Suburb']
                                != 'HILLSIDE (BRIMBANK)']
# Outlier for very high growth rate
house_price_df = house_price_df[house_price_df['Suburb'] != 'CARDIGAN']
# Outlier for very high growth rate
house_price_df = house_price_df[house_price_df['Suburb'] != 'BONNIE BROOK']


# Clear all non-numeric characters, set as float dtype
house_price_df = dataframe_tidyup(house_price_df)
unit_price_df = dataframe_tidyup(unit_price_df)


def calc_growth_rate_if_missing(df: pd.DataFrame, growth_col: int) -> pd.DataFrame:
    """
    Function to calculate missing growth rates in a DataFrame by iterating through yearly median house/unit price rows
    and columns, and using the oldest non-NaN property value to compare with the most recent property value calculate the growth rate.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to process.
    growth_col : int
        The column index of the column containing the growth rates.

    Returns
    -------
    DataFrame
        The processed DataFrame with missing growth rates filled in.
    """
    # Subset the df such that we have a df containing only the rows with a grwoth rate of NaN
    df_missing_growth_rates: pd.DataFrame = df[df.iloc[:, growth_col].isna()]
    # Iterate through the row of each suburb with a missing growth rate
    for row in range(len(df_missing_growth_rates)):
        # Iterate through 2013-2023 house/unit price values in each columns of the current suburb row
        for col_index in range(df_missing_growth_rates.shape[1]):
            oldest_value = df_missing_growth_rates.iloc[row,
                                                        col_index+1].item()
            # Check if the oldest_value is a float (and not NaN)
            if pd.notna(oldest_value) and isinstance(oldest_value, float):
                # Store the most recent median price so that we can compare it to the oldest value
                recent_price = df_missing_growth_rates.iloc[row, growth_col-1]
                # Calculate the number of years between the oldest value and the most recent value
                years_between = growth_col-1-col_index-1
                # Calculate the growth rate using the formula for CAGR
                calculated_growth_val = (
                    recent_price/oldest_value)**(1/years_between)
                # Write the calculated growth rate to the relevant cell in the dataframe
                df_missing_growth_rates.iloc[row, growth_col] = (
                    calculated_growth_val-1)*100
                break
    # Subset the original dataframe to those missing growth rates, and make it equal to the dataframe with the newly calculated growth rates
    df[df.iloc[:, growth_col].isna()] = df_missing_growth_rates
    return df


# Calculate the growth rate if it's missing
house_price_df = calc_growth_rate_if_missing(house_price_df, growth_col)
unit_price_df = calc_growth_rate_if_missing(unit_price_df, growth_col)


def extreme_growth_rate_check(df: pd.DataFrame, growth_col: int, max_growth_rate: float) -> pd.DataFrame:
    """
    Function to check for extreme changes in property prices over the first 3 years of 
    the data by examining each suburbs property prices, looking to see if there is a >50% increase in value
    between any of the first 3 years of data, and calculating the adjusted growth rate if an outlier is found.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to process.
    growth_col : int
        The column index of the column containing the growth rates.

    Returns
    -------
    DataFrame
        The processed DataFrame with missing growth rates filled in.
    """

    # Loop through all suburbs
    for row in range(len(df)):
        # Loop through all years of data for a suburb
        for col in range(df.shape[1]-5):
            # Check if the next 3 years of data are available (i.e. if there are values in the next 3 columns). If not available, move forward one year
            if all(pd.notna([df.iloc[row, col+1], df.iloc[row, col+2], df.iloc[row, col+3]])):
                # Check whether the first year of data is an outlier, based on whether it increases in value by >50% from the first to second year of available data
                if df.iloc[row, col+2]/df.iloc[row, col+1] > max_growth_rate:
                    # If there is a big increase to the second year, use the second year of data compared to the final year to calculate the growth rate
                    df.iloc[row, growth_col] = ((
                        df.iloc[row, growth_col-1]/df.iloc[row, col+2])**(1/(growth_col-col-3))-1)*100
                # Similarly, check whether the second year of data is an outlier, based on whether it increases in value by >50% from the second to third year of available data
                elif df.iloc[row, col+3]/df.iloc[row, col+2] > max_growth_rate:
                    # If there is a big increase to the third year, use the third year of data compared to the final year to calculate the growth rate
                    df.iloc[row, growth_col] = ((
                        df.iloc[row, growth_col-1]/df.iloc[row, col+3])**(1/(growth_col-col-4))-1)*100
                break
    return df


# Modify any growth rates that have outlier data affecting them
house_price_df = extreme_growth_rate_check(house_price_df, growth_col, 1.5)
unit_price_df = extreme_growth_rate_check(unit_price_df, growth_col, 1.5)

# Perform a join of the house and unit price data on the suburbs using only the relevant columns
unit_house_df: pd.DataFrame = pd.merge(house_price_df[['Suburb', 'H_2023_Prop_Price', 'H_Annual_Growth_Rate']], unit_price_df[['Suburb', 'U_2023_Prop_Price', 'U_Annual_Growth_Rate']],
                                       how='outer', on='Suburb', validate='one_to_one')


##########################################################################
#################### Calculate financial returns #########################
##########################################################################

# Initial Conditions
current_portfolio_value: float = 500000
interest_rate: dict[float, float] = {0.1: 0.0699,
                                     0.2: 0.064,
                                     0.3: 0.0574,
                                     0.4: 0.0564,
                                     0.5: 0.0559,
                                     0.6: 0.0559,
                                     0.7: 0.0559,
                                     0.8: 0.0559,
                                     0.9: 0.0559,
                                     1: 0.0559}
duration: int = 30
# after tax and expenses, but not rent
combined_discretionary_income: int = 120000
available_annual_mortgage_repayments: float = 2/3 * \
    combined_discretionary_income  # based on vague lending criteria found online
annual_cpi: float = 1.03
stock_return: float = 1.1

# Add columns to the dataframe to store whether the property is affordable, and the portfolio value after the duration


columns_to_insert: list[tuple[int, str, object]] = [
    (3, 'Is_house_affordable?', 'NaN'),
    (4, f'H+port_value_at_{duration}_years', np.nan),
    (7, 'Is_unit_affordable?', 'NaN'),
    (8, f'U+port_value_at_{duration}_years', np.nan)
]
for loc, col, val in columns_to_insert:
    unit_house_df.insert(loc=loc, column=col, value=val)


def calculate_portfolio_values(df: pd.DataFrame):
    """
    Function to calculate the portfolio values for each suburb for both houses and units, 
    determining whether the property is affordable based on available annual mortgage repayments,
    and calculating the portfolio value after a specified duration.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing suburb data.

    Returns
    -------
    None
        The function modifies the DataFrame in place.
    """
    # Loop through each suburb
    for suburb in range(df.shape[0]):
        # Complete all calcs for house data, then repeat for unit data
        for house_or_unit_binary_value in range(2):
            # Stop if NaN value
            if pd.notna(df.iloc[suburb, 1+house_or_unit_binary_value*4]):
                # Values at time = 0 for buying the property
                # Combined discretionary income resets each time
                combined_discretionary_income = 120000
                # Property value
                median_property_value = df.iloc[suburb,
                                                1+house_or_unit_binary_value*4]
                # Property growth rate
                median_property_growth_rate = df.iloc[suburb,
                                                      2+house_or_unit_binary_value*4]
                # Amount required to borrow
                borrowing_amount = float(
                    median_property_value) - float(current_portfolio_value)

                # If you don't need to borrow money, the surplus funds you have are converted to stock portfolio
                if borrowing_amount < 0:
                    stock_value = -borrowing_amount
                    required_annual_mortgage_repayments = 0

                # If you do need to borrow money, this calculates a) what interest rate you use for the given LVR,
                #  and b) the yearly mortgage repayments
                else:
                    stock_value = 0
                    # Deposit percent/LVR calculated to the nearest 10%, to be used with the dictionary describing interest rates for a given LVR
                    deposit_percent_rounded = math.ceil(
                        10*(current_portfolio_value/median_property_value))/10
                    # Annual mortgage repayments required for property
                    required_annual_mortgage_repayments = borrowing_amount*((interest_rate[deposit_percent_rounded])*(
                        1+interest_rate[deposit_percent_rounded])**duration)/(((1+interest_rate[deposit_percent_rounded])**duration)-1)

                # Can you afford to buy in this suburb?
                if required_annual_mortgage_repayments < available_annual_mortgage_repayments:
                    df.iloc[suburb, 3+house_or_unit_binary_value*4] = 'yes'
                    # Calculate the portfolio value after the specified duration
                    for _ in range(duration):
                        # Calculate annual property growth
                        median_property_value = (median_property_value) * \
                            (1+median_property_growth_rate/100)
                        # Calculate annual wage increases from CPI
                        combined_discretionary_income = combined_discretionary_income * \
                            (annual_cpi)
                        # Calculate remaining discretionary income to be used for stocks after mortgage repayments
                        stock_value = (stock_value+combined_discretionary_income -
                                       required_annual_mortgage_repayments)*(stock_return)
                    # Store the final portfolio result
                    df.iloc[suburb, 4 +
                            house_or_unit_binary_value*4] = float(stock_value) + float(median_property_value)
                else:
                    # Property is unaffordable
                    df.iloc[suburb, 3+house_or_unit_binary_value*4] = 'no'
                    df.iloc[suburb, 4+house_or_unit_binary_value *
                            4] = np.nan
    return df


unit_house_df = calculate_portfolio_values(unit_house_df)

###############################################################
#################### Generate the map #########################
###############################################################

# Convert the portfolio values to millions for easier reading on the map
unit_house_df[['H+port_value_at_30_years', 'U+port_value_at_30_years']
              ] = unit_house_df[['H+port_value_at_30_years', 'U+port_value_at_30_years']]/1000000

# Load the GeoJSON data
with open(r"C:\Users\leroy\Downloads\suburb-2-vic.geojson", 'r') as file:
    suburb_gdf = json.load(file)

# Function that adds data to the geojson file for each suburb


def add_data_to_geojson(geojson_data: dict[str, int], df: pd.DataFrame, list_of_columns_to_add: dict[str, int]):
    """
    Iterate through each suburb in the GeoJSON data and add the relevant data from the DataFrame to the geojson properties.

    Parameters
    ----------
    geojson_data : dict[str,int]
        The GeoJSON data to be processed.
    df : DataFrame
        The DataFrame containing the data to be added to the GeoJSON properties.
    list_of_columns_to_add : dict[str, int]
        A dictionary where the keys are the column names to be added and the values are the column indices in the DataFrame.

    Returns
    -------
    dict[str,int]
        The processed GeoJSON data with the added properties.
    """

    # Iterate through each suburb in the GeoJSON data
    for location in geojson_data['features']:
        # Initialize the new properties of the GeoJSON dict with default values
        for key in list_of_columns_to_add.keys():
            location["properties"][key] = "No data"
            location["properties"][f'{key}_growth_rate'] = "No data"
        # Match the suburb in the GeoJSON data with the suburb in the DataFrame
        for suburb in df['Suburb']:
            if location["properties"]["vic_loca_2"] == suburb:
                # Where they match add the relevant data to the geojson properties
                for key, value in list_of_columns_to_add.items():
                    # If the property is affordable, add the portfolio value, otherwise add "unaffordable"
                    if df[df['Suburb'] == suburb].iloc[0, value-1] == 'yes':
                        location["properties"][key] = F'{df[df['Suburb']
                                                            == suburb].iloc[0, value].astype('float64'):.2F}M'
                    elif df[df['Suburb'] == suburb].iloc[0, value-1] == 'no':
                        location["properties"][key] = "Unaffordable"
                    # Add the growth rate for the property value
                    location["properties"][
                        F'{key}_growth_rate'] = F'{df[df['Suburb'] == suburb].iloc[0, value-2].astype('float64'):.2F}%'
                break


list_of_columns = {'house_value': 4,
                   'unit_value': 8}
add_data_to_geojson(suburb_gdf, unit_house_df, list_of_columns)

# Set colour bins equal to the specified quantiles of the total data (houses and units)
new_dataframe_for_quantiles = pd.concat(
    [unit_house_df['H+port_value_at_30_years'], unit_house_df['U+port_value_at_30_years']])
max_val, top_25, top_50, top_75, min_val = (new_dataframe_for_quantiles).quantile(
    [1, 0.975, 0.5, 0.025, 0])

# Modify the suburbs in the dataframe that are unaffordable to $1 below the minimum value of affordable suburbs
for key, value in {'Is_house_affordable?': 4, 'Is_unit_affordable?': 8}.items():
    # Iterate through each suburb
    for element in range(len(unit_house_df)):
        # Check the column that tells whether the property is affordable. If no, set the portfolio value to $1 below the minimum value
        if unit_house_df[key].iloc[element] == 'no':
            unit_house_df.iloc[element, value] = min_val - 0.00001

# Create a Folium map centered on the centre of victoria, approximately around the town of Seymour
m = folium.Map(
    location=[-37.02655, 145.13924],
    zoom_start=8)

# Repeat the following for both houses and units, adding the house layer first so that its legend is shown on the map
for key, value in {'H+port_value_at_30_years': ('Median_House_Data', True), 'U+port_value_at_30_years': ('Median_Unit_Data', False)}.items():
    # Generate and store a choropleth layer for the houses and units
    stored_graph = folium.Choropleth(
        geo_data=suburb_gdf,
        data=unit_house_df,
        name=value[0],
        columns=['Suburb', key],
        key_on='feature.properties.vic_loca_2',
        fill_color='RdYlGn',
        legend_name=F'Median Property and Portfolio Value per Suburb in {duration} years time',
        bins=[min_val-0.00001, min_val, top_75, top_50, top_25, max_val],
        overlay=True,
        control=True,
        show=value[1]
    )
    # Add the house layer first so that its legend is shown on the map
    if key == 'H+port_value_at_30_years':
        stored_graph.add_to(m)
    else:
        # Delete the legend from the unit_layer layer prior to adding to m
        for key in list(stored_graph._children.keys()):
            if key.startswith('color_map'):
                del stored_graph._children[key]
        stored_graph.add_to(m)


# Add the layer for the popups and tooltip, with no style so that it's invisible
def style_function_popups(feature):
    return {
        'fillOpacity': 0,  # Make the GeoJson layer transparent
        'weight': 0  # No border
    }


popups = folium.GeoJson(
    suburb_gdf,
    name='Popups',
    style_function=style_function_popups,
    tooltip=folium.GeoJsonTooltip(fields=['vic_loca_2'], aliases=[
        'Suburb']),
    popup=folium.GeoJsonPopup(fields=['vic_loca_2', 'house_value', 'house_value_growth_rate', 'unit_value', 'unit_value_growth_rate'], aliases=[
        'Suburb', 'Portfolio and House Value', 'House growth per year', 'Portfolio and Unit Value', 'Unit growth per year'], labels=True, localize=True, parse_html=False),
    control=False
).add_to(m)

# Ensure the popups are on top of the choropleth/suburb layers
m.keep_in_front(popups)

# Create title to be added to html
map_title = f"PPOR + Market Index Investing Returns by Suburb in {duration} Years - Test"

# Create HTML for the title
title_html = f'''
             <h3 align="center" style="font-size:20px; color:black;"><b>{map_title}</b></h3>
             <h4 align="center" style="font-size:14px; color:black; font-style:italic; font-weight:bold">
                    Investment returns percentile colourscale:
                    <span style="font-size:14px; color:#CF6363; font-style:italic; font-weight:bold">
                    Red = Unaffordable,
                    </span>
                    <span style="font-size:14px; color:#F8C89F; font-style:italic; font-weight:bold">
                    Orange = 0 - 2.5%,
                    </span>
                    <span style="font-size:14px; color:#F1F4CA; font-style:italic; font-weight:bold">
                    Yellow = 2.5 - 50%,
                    </span>
                    <span style="font-size:14px; color:#C5D079; font-style:italic; font-weight:bold">
                    Green = 50 - 97.5%,
                    </span>
                    <span style="font-size:14px; color:#6CB57A; font-style:italic; font-weight:bold">
                    Dark Green = 97.5 - 100%
                    </span>
             </h4>
             '''

# Add the HTML title to the map's root element
m.get_root().html.add_child(folium.Element(title_html))

# Add layer control
folium.LayerControl().add_to(m)

m.save(r'C:\Users\leroy\OneDrive\Documents\VS Code Projects\choropeth_map.html')
