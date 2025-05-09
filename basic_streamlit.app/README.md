## Application Overview: Palmer's Penguins Explorer

This interactive web application enables users to explore the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/) through a streamlined interface for filtering, analyzing, and visualizing key biological traits of penguin species. Built using Streamlit, the app provides a clean, intuitive way to perform exploratory data analysis.

### Features

* **Data Filtering**

  * Users can filter the dataset by:

    * Penguin species
    * Island location
    * Sex
    * Body mass (grams) via a dynamic range slider

* **Summary Statistics**

  * A statistical overview is generated based on the selected filters, including measures of central tendency and distribution for numeric variables

* **Interactive Data Table**

  * A scrollable and filter-responsive table displays the full subset of filtered observations

* **Visualizations**

  * **Body Mass Histogram**: A distribution plot (with optional KDE) shows the range and frequency of body mass values for selected penguins
  * **Scatter Plot of Flipper Length vs. Body Mass**: Highlights relationships between body size characteristics across different species and sexes

---

## How to Use the Application

### Requirements

To run the application locally, install the following dependencies:

* `streamlit`
* `pandas`
* `seaborn`
* `matplotlib`

You can install all dependencies by running:

```bash
pip install streamlit pandas seaborn matplotlib
```

### Running the Application

1. Clone the repository or download the application code
2. Navigate to the project directory
3. Run the Streamlit application with the following command:

```bash
streamlit run app.py
```

Make sure the `penguins.csv` dataset is located in a `data/` folder relative to the application script.

---

## Dataset Information

The dataset used in this application is the Palmer Penguins dataset, developed by Dr. Kristen Gorman and made publicly available through the [palmerpenguins R package](https://github.com/allisonhorst/palmerpenguins). It includes biometric data for three penguin species observed in the Palmer Archipelago, Antarctica.

### Key Attributes:

* `species`: Penguin species (Adelie, Chinstrap, Gentoo)
* `island`: Island where the penguin was observed (Biscoe, Dream, Torgersen)
* `sex`: Male or female
* `body_mass_g`: Body mass in grams
* `flipper_length_mm`: Flipper length in millimeters
