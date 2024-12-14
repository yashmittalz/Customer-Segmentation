# Advanced Customer Segmentation using Machine Learning

## Overview

Customer segmentation assumes that different customers require different marketing programs—different offers, prices, promotions, distribution methods, or a combination of these variables. This project focuses on segmenting customers into meaningful groups using advanced clustering techniques to assist businesses in crafting targeted marketing strategies.

## Project Goals

- Implement advanced clustering algorithms to segment customers effectively.

- Handle complex datasets containing both categorical and numerical data.

- Provide insightful visualizations to interpret customer clusters.

## Predictive Examples

By analyzing customer data, this project enables predictions such as:

- Identifying high-value customer segments.

- Tailoring marketing campaigns for different clusters.

- Recommending products or services based on customer segment behaviors.

## Features

- Supports mixed datasets with categorical and numerical variables.

- Utilizes the KPrototypes algorithm from the Kmodes library.

- Provides robust visualizations for cluster analysis.

## Requirements

- Python 3.8 or above

- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `kmodes`

To install the required libraries, run:
```bash 
    pip install -r requirements.txt 
```

Usage

Clone the repository:
```bash
    git clone <https://github.com/yashmittalz/Customer-Segmentation>
    cd <https://github.com/yashmittalz/Customer-Segmentation>
```

Install dependencies:
```bash
    pip install -r requirements.txt
```

Run the application:
```bash
    python main.py
```

Follow the menu prompts to load data, preprocess, cluster, and visualize.

## Dataset

The dataset used in this project contains customer-related information with both categorical and numerical data. It is available on Kaggle and can be downloaded here:
[Retail Dataset Analysis](https://www.kaggle.com/datasets/khalidnasereddin/retail-dataset-analysis)

Place the dataset in the same directory as the project files and name it segmentation-data.csv.

## Project Background

### Why Clustering?

Clustering is essential for customer segmentation as it allows businesses to:

- Establish better customer relationships by understanding their unique needs.

- Focus on the most profitable customer groups.

- Upsell and cross-sell effectively by identifying preferences within segments.

- Develop more efficient communication strategies tailored to each segment.

### Why MinMaxScaler?

The MinMaxScaler is used to normalize numerical features to a uniform scale (0 to 1). This is crucial because clustering algorithms are sensitive to the range of data, and normalization ensures that all features contribute equally to the clustering process.

### Why KPrototypes?

Traditional clustering algorithms like KMeans are limited to numerical data. The KPrototypes algorithm from the Kmodes library supports mixed data types, making it ideal for our dataset, which contains both categorical and numerical variables.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Yash Mittal. Version 1.0

*Date Created: November 24, 2024*
*Last updated: December 13, 2024*
