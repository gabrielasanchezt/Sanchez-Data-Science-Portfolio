# Tidy Data Project: Athlete Medal Analysis

This project focuses on transforming and analyzing a dataset containing information about 2008 Olympic medalists. The goal is to apply tidy data principles to restructure a complex dataset into a clean, analysis-ready format, followed by visual and tabular exploratory data analysis.

## Project Overview

The dataset includes information about athletes, their sports, and the medals they won. The data is initially in a wide format, with columns combining gender, sport, and medal type. Using tidy data principles, this project reshapes the dataset so that each variable has its own column, each observation is a row, and each type of observational unit forms its own table.

The project involves:

* Data cleaning using pandas, including handling missing values and renaming columns for clarity
* Reshaping the dataset using `melt()` to convert it to a long format
* Splitting combined column values into meaningful separate variables (e.g., sport and medal type)
* Filtering and summarizing key patterns in the data
* Creating visualizations that highlight medal distribution by athlete and gender

## Tidy Data Principles Applied

* Each **variable** is stored in its own column
* Each **observation** is stored in its own row
* Each **type of observational unit** is stored in its own table

These principles were used to transform a non-tidy dataset into a structured and interpretable format suitable for analysis.

## Setup and Usage

### Prerequisites

To run this project, the following Python libraries are required:

* pandas
* matplotlib
* seaborn
* Jupyter Notebook

To install all required libraries, run:

```bash
pip install pandas matplotlib seaborn notebook
```

### Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/gabrielasanchezt/Sanchez-Data-Science-Portfolio.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Sanchez-Data-Science-Portfolio/Tidy_Data
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open the notebook file named `Tidy-Data-Project.ipynb` and run all cells sequentially.

## Dataset Description

The dataset includes the following features:

* `medalist_name`: Name of the athlete
* Columns for each gender-sport combination (e.g., `male_archery`, `female_athletics`), where the value indicates the medal won (gold, silver, or bronze)

After preprocessing, the dataset contains:

* `medalist_name`
* `sport`: Extracted from the original column name
* `gender`: Extracted from the original column name
* `medal`: The medal type awarded

## Data Cleaning and Transformation

Key transformation steps included:

* Handling missing values
* Melting the dataset from wide to long format
* Splitting combined column values (e.g., `female_archery`) into separate `gender` and `sport` columns
* Renaming columns to be consistent and descriptive
* Filtering for top athletes by total medal count
* Creating a pivot table to aggregate medal counts by athlete and medal type

## Visualizations

Two primary visualizations were created:

1. **Total Medal Count by Athlete**
   A bar chart showing the number of gold, silver, and bronze medals for the top 10 athletes by total medal count.

2. **Medal Distribution by Gender**
   A grouped bar chart visualizing the distribution of gold, silver, and bronze medals by gender.

Images are located in the `images/` directory and linked within the notebook.

## Sources and References

* Dataset Source: Olympic Medalists Dataset, 2008
* [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
* [Tidy Data (Hadley Wickham)](https://vita.had.co.nz/papers/tidy-data.pdf)

