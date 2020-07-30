# Airbender Project

For those who loves travelling and uses Airbnb a lot, do you ever have difficulties to find the best listings? I do! Hence, Airbender project, Airbnb Listings Recommender System.

There are some questions when I access Airbnb:
- "I’ve never been to this city, but I know what kind of place I like to stay!"
- "I stayed once here, and I love the listing! Where I can find other similar listings?"
- "I came to this city a lot. I gave comment to those listings and others too. Where do they usually stay?"

Based on those problems, I decided to build a recommendation based on:
1. User preference
2. Content similarity
3. Preferences from many users



## Prerequisites

Here are some packages needed in addition to default python packages. Please install at your own risk.


wordcloud package for visualization of the text:
```
conda install -c conda-forge wordcloud
```

scikit-learn as machine learning tools:
```
pip install -U scikit-learn
```

NLTK package for NLP process:
```
pip install --user -U nltk
```

Textblob package for sentiment analysis:
```
conda install -c conda-forge textblob
```

Surprise library for collaborative modeling:
```
conda install -c conda-forge scikit-surprise
```



## Data Description

The datasets are from insideairbnb.com. They are categorized by city and I used data from Vancouver. There are 4 datasets in the folder as initial data source, but it ended up with 2 datasets for this project:
- listings.csv : This file contains all information about the listings in Vancouver, such as the name, description, summary, price, accommodation, etc.
- reviews.csv : This dataset represents the review for each listing from different users. It has listing id, user id, some other columns, and most importantly, content of the review.



## Method Used for Recommender System

- Content-based filtering : to tackle the first 2 problems, recommendation from user query and content similarity
- Collaborative filtering : to solve the third issue, recommendation from preferences by other similar users



## Directory and File Information

- `data` folder : store all raw datasets from insideairbnb.com
- `download` folder : store all files that were downloaded from the notebooks, for instance: pkl files and processed csv files 
- `1_Introduction_and_Content_Based.ipynb` :  First notebook, data analysis and first recommender algorithm (Content-Based)
- `2_Collaborative.ipynb` : Second notebook, collaborative filtering method #####without##### modeling
- `3_Collaborative_with_Modelling` : Third notebook, collaborative filtering #####with##### modeling using Surprise library
- `Listing_content_packed_bubbles.twb` : Tableau workbook for listings content word visualization
- `airbender.py` : Python file for the streamlit apps
- `report.pdf` : Complete report of the project
- `vancouver.jpeg` : Image for the Streamlit apps homepage



## How to Run the Streamlit Apps

It is suggested to use a new Python environment for the Streamlit app (for instance: mystream).
```
conda create --name mystream python
```

Install all the basic packages, such as numpy, pandas, scikit-learn.

Install the Streamlit in this environment.
```
pip install streamlit
```

Install streamlit-folium and folium for maps.
```
conda install -c conda-forge folium
pip install streamlit-folium
```

Run the Python file in this environment. Make sure the terminal is in the correct folder of the file.
```
streamlit run airbender.py
```

## END


