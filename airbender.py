#Import all the required packages
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import folium
from streamlit_folium import folium_static

#Set the title of the apps
st.title('AirBender Project')

#Set the header of sidebar
st.sidebar.header('Navigation')

#Set the radio button for home page and 3 different users
user_type = st.sidebar.radio("Go to:", ('Preface', 'New User', 1969, 11967216))

#Load all pkl files from the notebook
tfidf_content = joblib.load("download/tfidf_content.pkl")    
df_listings = joblib.load("download/df_listings.pkl")    
vectorizer = joblib.load("download/vectorizer.pkl")    
df_user = joblib.load("download/df_user.pkl")  
keys = joblib.load("download/keys.pkl")  
top_n_SVD = joblib.load("download/top_n_SVD.pkl")

#Function to get the vectorized data of the content from the key (similar in the notebook)
def get_listing_by_id(id, tfidf_content, keys):
    row_id = keys[id]
    row = tfidf_content[row_id,:]
    return row

#Function to get the recommendation data for content-based
def content_recommender_fin(tfidf_content, listings, num, listing_ids = None, user_preferences = None):

    """Return num recommendation based on listing_ids or/and user_preferences

    Args:
        tfidf_content : dataframe of vectorized listing content
        listings : dataframe of listing, to get the listing information for the output
        num : total recommendation for the output
        listing_ids : list of listing_ids to be compared to all listings
        user_preferences : string of user query

    Returns:
    A dataframe of listings recommendations with total rows of num
    """

    #Set dataframe variable for the result
    similar_listings = pd.DataFrame(columns = ["neighborhood", "price", "name", "summary", "space", "description", 
                                               "neighborhood_overview", "notes", "transit", "access", "house_rules", 
                                               "similarity", "id", "listing_url", "latitude", "longitude"] )
    
    if user_preferences != None:
        #Get the transform data of user preferences information
        user_pref_transform = vectorizer.transform([user_preferences])
    
    if listing_ids != None:
        #Set list for exclude the sample listing_ids index for similar_listings
        my_listing_indexes = []

        #Set dataframe for all sample listings
        my_listings = pd.DataFrame()

        #Get sample listings data and index
        for listing_id in listing_ids:
            my_listings = vstack((my_listings, get_listing_by_id(listing_id, tfidf_content, keys)))
            my_listing_indexes.append(keys[listing_id])

        #If user pref is exists, add as 1 of sample data
        if user_preferences != None:
            my_listings = vstack((my_listings, user_pref_transform))
            
        #Set average value of each column of the sample listings
        my_listing = my_listings.mean(axis=0)
        
    else:
        #If no listing_ids, set my_listing with user pref only
        my_listing = user_pref_transform
    
    #Cosine similarity between the average of sample listings and all listings
    similarity = cosine_similarity(my_listing, tfidf_content)    
    
    #Loop from most similar data
    for index in similarity[0].argsort()[::-1][:]:
        
        #Exclude the same listing with our sample
        if listing_ids != None:
            if index in my_listing_indexes:
                continue
            
        #Append the data into dataframe
        similar_listings.loc[len(similar_listings)] = [listings.iloc[index].loc['neighbourhood_cleansed'],
                                                       listings.iloc[index].loc['price'], 
                                                       listings.iloc[index].loc['name'], 
                                                       listings.iloc[index].loc['summary'], 
                                                       listings.iloc[index].loc['space'], 
                                                       listings.iloc[index].loc['description'],
                                                       listings.iloc[index].loc['neighborhood_overview'], 
                                                       listings.iloc[index].loc['notes'], 
                                                       listings.iloc[index].loc['transit'], 
                                                       listings.iloc[index].loc['access'], 
                                                       listings.iloc[index].loc['house_rules'], 
                                                       similarity[0][index],
                                                       listings.iloc[index].loc['id'], 
                                                       listings.iloc[index].loc['listing_url'], 
                                                       listings.iloc[index].loc['latitude'],
                                                       listings.iloc[index].loc['longitude'],
                                                       ]
        
        #Break from the loop when reach the num
        if len(similar_listings) == num:
            break

    return similar_listings

#Function to show map
def show_map(df_fin):
    folium_map = folium.Map(location=[49.286310, -123.134108], zoom_start=12, height='65%')

    for index,row in df_fin.iterrows():
        popup = '<a href="' + row["listing_url"] + '"target="_blank"> Visit! </a>'
        folium.Marker([row["latitude"], row["longitude"]], popup=popup, tooltip=row["name"]).add_to(folium_map)

    return folium_static(folium_map)

#Set empty data frame for the result of recommendation function
df_fin = pd.DataFrame()

#If Preface, then show information for home page
if user_type == 'Preface':
    st.image('vancouver.jpg', width=700)
    """
    ## Welcome to Vancouver Airbnb Listings Recommender System!

    This page is to show case my capstone project to recommend listings at Airbnb. There are two methods for recommendation, which are 
    content based and collaborative filtering. Below are some guidelines on how to navigate this page:
    
    - 'New User' : As a new user, there is no history of stay at Airbnb in Vancouver. But we can still give some recommendation based 
                   on user preference. User can entry some preference in the text area and system will show some listings that similar
                   with the preferences.
                   
    - '1969'     : This is a user who stayed one time at a listing in Vancouver. We can see that listing, other listing that similar to
                   the listing, and user still can do some entry for their own preference to find different type of listings.
                   
    - '11967216' : This is a user who has stayed at multiple listings in Vancouver. In addition to some features that we have for previous
                   user, here we also can see some recommendation based on other users who also ever stayed in the listings and have
                   common comments. This is the collaborative filtering. We can't see the same feature for previous user because he/she
                   only stayed in one listing and we could't find the similarity with other users.
                   
    Thank you for visiting my web page!
    """
    #Show empty Vancouver map in the homepage
    folium_map = folium.Map(location=[49.286310, -123.134108], zoom_start=12, height='65%')
    folium_static(folium_map)

#If New User:
elif user_type == 'New User':
    #Put text box for user query
    user_preferences = st.sidebar.text_area("Please let me know your listing preference!")

    if user_preferences != '':
        #Set the subheader
        st.subheader(f'Your preference listings:')

        #Get the recommendation from user preference and show in the map
        df_fin = content_recommender_fin(tfidf_content, df_listings,5, user_preferences = user_preferences)
        st.dataframe(df_fin)
        show_map(df_fin)

#If not a new user:
else:

    #Get the listing id from df_user based on selected user_id
    listing_ids = list(df_user[df_user['reviewer_id'] == user_type]['listing_id'])

    #Set the radio button for output category
    show = st.sidebar.radio("Show me:", ('My previous stays', 'Similar listings with my previous stays', 'Listing from other travelers', 'My preferences'))

    #If output category is previous stays:
    if show == 'My previous stays':

        #Show traveler previous stays
        st.subheader(f'Your previous stay(s):')

        #Get data from listings dataframe based on listing ids
        df_prev_stay = df_listings[df_listings['id'].isin(listing_ids)][['neighbourhood_cleansed', 'price', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'house_rules', 'id', 'listing_url', 'latitude', 'longitude']]
        st.dataframe(df_prev_stay)
        show_map(df_prev_stay)


    #If output category is similar listings
    elif show == 'Similar listings with my previous stays':

        #Content recommendation from previous stays
        st.subheader(f'Similar listings with your previous stay(s):')

        #Get the recommendation based on listing ids
        df_content_rec = content_recommender_fin(tfidf_content, df_listings,5, listing_ids = listing_ids)
        st.dataframe(df_content_rec)
        show_map(df_content_rec)


    #If output category is listing from other travelers
    elif show == 'Listing from other travelers':

        #Collaborative recommendation
        st.subheader(f'Other traveler(s) who stays at your previous listings also like:')

        #Get the data from collaborative filtering result dictionary (top_n_SVD)
        listing_ids = [i[0] for i in top_n_SVD[user_type]]
        df_collab = df_listings[df_listings['id'].isin(listing_ids)][['neighbourhood_cleansed', 'price', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'house_rules', 'id', 'listing_url', 'latitude', 'longitude']]
        st.dataframe(df_collab)
        show_map(df_collab)

    #This is for user preference query for non new user
    else:

        #Set the text box
        user_preferences = st.sidebar.text_area("Please let me know your listing preference!")
        if user_preferences != '':
            st.subheader(f'Your preference listings:')

            #Get recommendation from user query
            df_fin = content_recommender_fin(tfidf_content, df_listings,5, user_preferences = user_preferences)
            st.dataframe(df_fin)
            show_map(df_fin)
        
            
    

    





