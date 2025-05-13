# Import required libraries
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
import calendar
import gdown
warnings.filterwarnings('ignore')



# Configure the Streamlit dashboard settings
st.set_page_config(page_title="Movies_Recommendation_System", page_icon=":bar_chart:", layout="wide")

# Apply custom CSS for layout styling
st.markdown("""
    <style>
        div.block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Display main dashboard title
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center;">
        <h1 style="font-family: 'Courier New', Courier, monospace; font-weight: bold; font-size: 60px; margin-right: 40px;">MovieLens Dashboard</h1>
    </div>
""", unsafe_allow_html=True)


# Add logo image to sidebar
st.sidebar.image("moviel__1_-removebg-preview.png", width=200) 

# Load dataset
file_id = "1QPIs-M-0Mc44hu_vGW5PuB-TCx7Ro7ca"
# Create the direct download URL
url = f"https://drive.google.com/uc?id={file_id}"
# Optional: specify the output filename
output = "data.csv"  # change extension if it's not CSV
gdown.download(url, output, quiet=False)

df = pd.read_csv("data.csv",nrows=5000)


# Sidebar header for filter section
st.sidebar.header("Choose your filter: ")

col1, col2= st.columns((2))

# Drop missing values and ensure movie_year is of integer type
df = df.dropna(subset=["movie_year"])
df["movie_year"] = df["movie_year"].astype(int)

# Keep only years > 0
df = df[df["movie_year"] > 0]

# Extract valid year ranges for filtering
years = sorted(df["movie_year"].unique())

# Remove the last year from start_year options
start_years = years[:-1] 
end_years = years         
# Year range filter inputs
with col1:
    start_year = st.selectbox("Select Start Year", start_years, index=0)

with col2:
    end_year = st.selectbox("Select End Year", end_years, index=len(end_years)-1)

# Year range validation
if start_year > end_year:
    st.warning("⚠️ Start year must be less than or equal to end year.")
    st.stop()  

# Filter the data based on selected years
df = df[(df["movie_year"] >= start_year) & (df["movie_year"] <= end_year)].copy()


# Filter dataset by user ratings
ratings = sorted(df["user_rating"].unique())
selected_ratings = st.sidebar.multiselect("Pick Rating(s)", ratings)
if selected_ratings:
    df1 = df[df["user_rating"].isin(selected_ratings)]
else:
    df1 = df.copy()

# Filter genres
genre_columns = [
    "(no genres listed)", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]
selected_genres = st.sidebar.multiselect("Pick Genre(s)", genre_columns)

if selected_genres:
    genre_filter = df1[selected_genres].sum(axis=1) > 0
    filtered_df = df1[genre_filter]
else:
    filtered_df = df1.copy()

# Compute initial KPIs (total movies, users, ratings)
total_movies = filtered_df['movieId'].nunique()
total_users = filtered_df['userId'].nunique()
total_ratings = filtered_df['user_rating'].count()
# Display KPI stat cards
st.markdown("""
    <style>
        .stat-card {
            border: 2px solid rgba(160, 160, 160, 0.3);  /* بوردر رصاصي شفاف */
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            background-color: rgba(200, 200, 200, 0.2);  /* كارت رصاصي شفاف */
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #FF8282;
        }
        .stat-title {
            font-size: 20px;
            color: "Black";
            text-align: center;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Total Number of Movies</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_movies}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Total Number of Users</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_users}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Total Number of Ratings</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_ratings}</div>
        </div>
    """, unsafe_allow_html=True)

# Divider line
st.markdown("""
    <style>
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
    <hr class="custom-hr">
""", unsafe_allow_html=True)


# Line chart: Movies released per year
col1, col2 = st.columns([2, 1]) 
unique_movies_df = filtered_df.drop_duplicates(subset='movieId')
unique_movies_df['movie_year'] = unique_movies_df['movie_year'].astype(str).str.extract('(\d{4})')[0]
movies_per_year = unique_movies_df['movie_year'].value_counts().sort_index(ascending=False).head(40)
movies_per_year = movies_per_year.sort_index()
movies_df = movies_per_year.reset_index()
movies_df.columns = ['movie_year', 'count']
fig = px.line(
    movies_df,
    x='movie_year',
    y='count',
    markers=True,
    labels={'movie_year': 'Year', 'count': 'Number of Movies'},
)

fig.update_traces(line=dict(color='#5ce1e6', width=1.5), marker=dict(size=4))
fig.update_layout(
    shapes=[
        dict(
            type='rect',
            xref='paper', yref='paper',
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color='rgba(160,160,160,0.5)', width=2),
            layer='below'
        )
    ],
    plot_bgcolor='rgba(255,255,255,0.1)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    xaxis_tickangle=-45,
    showlegend=False,
    margin=dict(l=10, r=10, t=5, b=5),
    height=350,  
)

with col1:
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Movies Released Per Year</h3>",
        unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)
 
# Donut charts: Movies per genre and ratings distribution
with col2:
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Movies per Genre</h3>",
        unsafe_allow_html=True
    )
    genre_cols_to_plot = selected_genres if selected_genres else genre_columns
    genre_counts = filtered_df.drop_duplicates(subset='movieId')[genre_cols_to_plot].sum()
    genre_counts = genre_counts[genre_counts > 0]
    total = genre_counts.sum()
    labels_with_percent = [f"{genre} ({value / total:.1%})" for genre, value in genre_counts.items()]

    fig1 = px.pie(
        values=genre_counts.values,
        names=labels_with_percent,
        hole=0.6
    )
    fig1.update_traces(textinfo='none')  
    fig1.update_layout(
        height=130, 
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="v", 
            y=0.5, 
            x=1.1,  
            traceorder='normal', 
            font=dict(size=10),  
            title="Genres"
        )
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Ratings Distribution</h3>",
        unsafe_allow_html=True
    )

    rating_counts = filtered_df['user_rating'].value_counts().sort_index()
    total = rating_counts.sum()
    labels_with_percent = [f"{rating} ({value / total:.1%})" for rating, value in rating_counts.items()]
    fig2 = px.pie(
        values=rating_counts.values,
        names=labels_with_percent,
        hole=0.6
    )

    fig2.update_traces(textinfo='none')  
    fig2.update_layout(
        height=130, 
        margin=dict(t=10, b=10, l=0, r=150),
        legend=dict(
            orientation="v", 
            y=0.5, 
            x=1.1,  
            traceorder='normal', 
            font=dict(size=10), 
            title="Ratings"
        )
    )
    st.plotly_chart(fig2, use_container_width=False)

st.markdown("""
    <style>
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
    <hr class="custom-hr">
""", unsafe_allow_html=True)


# Bar charts: Top and bottom rated movies
def split_title(title, max_words_per_line=5):
    words = title.split()
    lines = [' '.join(words[i:i+max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
    return '<br>'.join(lines)

top_movies = filtered_df.sort_values(by="movies_avg_rating", ascending=False).drop_duplicates("movie_title").head(10)
bottom_movies = filtered_df.sort_values(by="movies_avg_rating", ascending=True).drop_duplicates("movie_title").head(10)

top_movies['formatted_title'] = top_movies['movie_title'].apply(split_title)
bottom_movies['formatted_title'] = bottom_movies['movie_title'].apply(split_title)

col1, col2 = st.columns((2))
with col1:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Top Rated Movies</h3>", unsafe_allow_html=True)
    
    fig_top = px.bar(
        top_movies.sort_values("movies_avg_rating"),
        x="movies_avg_rating",
        y="formatted_title",  
        orientation='h',
        color="movies_avg_rating",
        color_continuous_scale="teal",
        labels={"movies_avg_rating": "Average Rating", "formatted_title": "Movie Title"}
    )
    
    fig_top.update_layout(
        height=400,  
        yaxis_title="", 
        xaxis_title="Average Rating", 
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickmode='array', tickvals=top_movies['formatted_title'].tolist(), ticktext=top_movies['formatted_title'].tolist()), # محاذاة النص
        shapes=[
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='rgba(160,160,160,0.5)', width=2),
                layer='below'
            )
        ]
    )
    
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-family: \"Courier New\", Times, serif;'>Bottom Rated Movies</h3>", unsafe_allow_html=True)
    fig_bottom = px.bar(
        bottom_movies.sort_values("movies_avg_rating"),
        x="movies_avg_rating",
        y="formatted_title",
        orientation='h',
        color="movies_avg_rating",
        color_continuous_scale="RedOr",
        labels={"movies_avg_rating": "Average Rating", "formatted_title": "Movie Title"}
    )
    
    
    fig_bottom.update_layout(
        height=400,  
        yaxis_title="", 
        xaxis_title="Average Rating", 
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickmode='array', tickvals=bottom_movies['formatted_title'].tolist(), ticktext=bottom_movies['formatted_title'].tolist()), # محاذاة النص
        shapes=[
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='rgba(160,160,160,0.5)', width=2),
                layer='below'
            )
        ]
    )
    
    st.plotly_chart(fig_bottom, use_container_width=True)

# Divider line
st.markdown("""
    <style>
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
    <hr class="custom-hr">
""", unsafe_allow_html=True)


# Line charts: Rating trend over years and count of each rating value

col1, col2 = st.columns([2, 1])  
 
with col1:
    st.markdown("<h3 style='text-align: center; font-size: 18px; font-family: \"Courier New\", Times, serif;'>Avg. Rating by Year</h3>", unsafe_allow_html=True)
    rating_by_year = filtered_df.dropna(subset=["movie_year", "user_rating"])
    rating_by_year = rating_by_year[rating_by_year["movie_year"] > 0]
    rating_by_year = rating_by_year.groupby("movie_year")["user_rating"].mean().reset_index()
    
    fig1 = px.line(
        rating_by_year,
        x="movie_year", 
        y="user_rating", 
        template="plotly_dark", 
        labels={"movie_year": "Year", "user_rating": "Average Rating"},
        markers=True
    )
    fig1.update_traces(line=dict(color='#FD8A8A', width=1.5), marker=dict(size=4))  # تخصيص الشكل
    fig1.update_layout(
        shapes=[ 
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='rgba(160,160,160,0.5)', width=1.5),
                layer='below'
            )
        ],
        plot_bgcolor='rgba(255,255,255,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(l=10, r=10, t=0, b=0),
        height=200,  
    )
    st.plotly_chart(fig1, use_container_width=True)


    st.markdown("<h3 style='text-align: center; font-size: 18px; font-family: \"Courier New\", Times, serif;'>Number of Ratings per Rating Value</h3>", unsafe_allow_html=True)
    rating_counts = filtered_df["user_rating"].value_counts().sort_index()
    rating_df = pd.DataFrame({"Rating": rating_counts.index, "Count": rating_counts.values})
    
    fig2 = px.line(
        rating_df, 
        x="Rating", 
        y="Count", 
        markers=True, 
        template="plotly_white"
    )
    fig2.update_traces(line=dict(color='#FF8282', width=1.5), marker=dict(size=4))  # تخصيص الشكل
    fig2.update_layout(
        shapes=[ 
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='rgba(160,160,160,0.5)', width=1.5),
                layer='below'
            )
        ],
        plot_bgcolor='rgba(255,255,255,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(l=10, r=10, t=0, b=0),
        height=180,  
    )
    st.plotly_chart(fig2, use_container_width=True)


# Horizontal bar chart: Average rating by genre
with col2:
    st.markdown("<h3 style='text-align: center; font-size: 18px; font-family: \"Courier New\", Times, serif;'>Genres Average Rating</h3>", unsafe_allow_html=True)
    if selected_genres:
        genres_to_plot = selected_genres
    else:
        genres_to_plot = [genre for genre in genre_columns if genre in filtered_df.columns]

    avg_ratings_by_tag = {}
    for tag in genres_to_plot:
        genre_df = filtered_df[filtered_df[tag] == 1]
        if not genre_df.empty:
            avg_rating = genre_df['user_rating'].mean()
            avg_ratings_by_tag[tag] = avg_rating

    avg_ratings_df = pd.DataFrame.from_dict(avg_ratings_by_tag, orient='index', columns=['avg_rating']).dropna()
    avg_ratings_df = avg_ratings_df.sort_values(by='avg_rating', ascending=True).reset_index()
    avg_ratings_df = avg_ratings_df.rename(columns={'index': 'Genre'})

    fig = px.bar(
        avg_ratings_df,
        x='avg_rating',
        y='Genre',
        orientation='h',
        color='avg_rating',
        color_continuous_scale='teal',
        labels={'avg_rating': 'Average Rating'},
    )

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)',
        yaxis_title=None,
        xaxis_title='Average Rating',
        title="",
        font=dict(family="Times New Roman", size=14),
        height=400,
        margin=dict(l=0, r=0, t=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    
# Divider line
st.markdown("""
    <style>
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
    <hr class="custom-hr">
""", unsafe_allow_html=True)

# Pie charts: Rating distribution by hour, year, month, weekday

st.markdown("<h3 style='text-align: center; font-size: 25px; font-family: \"Courier New\", Times, serif;'>Rating Distribution by Time</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

# Hourly rating distribution
with col1:
    hour_counts = filtered_df["hour"].dropna().value_counts().sort_index()
    if not hour_counts.empty:
        total = hour_counts.sum()
        labels_with_percent = [f"{hour} ({value / total:.1%})" for hour, value in hour_counts.items()]
        fig_hour = px.pie(
            values=hour_counts.values,
            names=labels_with_percent,
            hole=0.6,
            title="🕒 Hour",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_hour.update_traces(textinfo='none')  
        fig_hour.update_layout(
            template="plotly_dark",
            title_font_size=14,
            height=200, 
            margin=dict(t=30, b=0, l=5, r=5),  
            legend=dict(
                orientation="v",
                y=0.5,
                x=1.1,
                traceorder='normal',
                font=dict(size=10),
                title="Hour"
            )
        )
        st.plotly_chart(fig_hour, use_container_width=False)

# Yearly rating distribution
with col2:
    year_counts = filtered_df["year"].dropna().astype(int).value_counts().sort_index()
    if not year_counts.empty:
        total = year_counts.sum()
        labels_with_percent = [f"{year} ({value / total:.1%})" for year, value in year_counts.items()]
        fig_year = px.pie(
            values=year_counts.values,
            names=labels_with_percent,
            hole=0.6,
            title="📅 Year",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_year.update_traces(textinfo='none')  
        fig_year.update_layout(
            template="plotly_dark",
            title_font_size=14,
            height=200, 
            margin=dict(t=30, b=0, l=5, r=5),
            legend=dict(
                orientation="v",
                y=0.5,
                x=1.1,
                traceorder='normal',
                font=dict(size=10),
                title="Year"
            )
        )
        st.plotly_chart(fig_year, use_container_width=False)

# Monthly rating distribution
with col3:
    month_counts = filtered_df["month"].dropna().astype(int).value_counts().sort_index()
    month_names = [calendar.month_name[m] for m in month_counts.index]
    if not month_counts.empty:
        total = month_counts.sum()
        labels_with_percent = [f"{month} ({value / total:.1%})" for month, value in zip(month_names, month_counts.values)]
        fig_month = px.pie(
            values=month_counts.values,
            names=labels_with_percent,
            hole=0.6,
            title="📆 Month",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_month.update_traces(textinfo='none')  
        fig_month.update_layout(
            template="plotly_dark",
            title_font_size=14,
            height=200, 
            margin=dict(t=30, b=0, l=5, r=5),
            legend=dict(
                orientation="v",
                y=0.5,
                x=1.1,
                traceorder='normal',
                font=dict(size=10),
                title="Month"
            )
        )
        st.plotly_chart(fig_month, use_container_width=False)

# Weekday rating distribution
with col4:
    if {"year", "month", "day"}.issubset(filtered_df.columns):
        filtered_df["weekday"] = pd.to_datetime(filtered_df[["year", "month", "day"]], errors='coerce').dt.day_name()
        weekday_counts = filtered_df["weekday"].dropna().value_counts().sort_index()
        if not weekday_counts.empty:
            total = weekday_counts.sum()
            labels_with_percent = [f"{weekday} ({value / total:.1%})" for weekday, value in weekday_counts.items()]
            fig_weekday = px.pie(
                values=weekday_counts.values,
                names=labels_with_percent,
                hole=0.6,
                title="📅 Weekday",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_weekday.update_traces(textinfo='none')  
            fig_weekday.update_layout(
                template="plotly_dark",
                title_font_size=14,
                height=200, 
                margin=dict(t=30, b=0, l=5, r=5),
                legend=dict(
                    orientation="v",
                    y=0.5,
                    x=1.1,
                    traceorder='normal',
                    font=dict(size=10),
                    title="Weekday"
                )
            )
            st.plotly_chart(fig_weekday, use_container_width=False)
    else:
        st.warning("Columns 'year', 'month', 'day' not found for weekday conversion.")

st.markdown("""
    <style>
        .custom-hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #F7418F, #FFCBCB, #3AA6B9);
            margin: 40px 0;
            border-radius: 5px;
        }
    </style>
    <hr class="custom-hr">
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; font-size: 25px; font-family: \"Courier New\", Times, serif;'>Number of Ratings vs. Average Rating per Movie</h3>", unsafe_allow_html=True)

# Final visualization: Ratings count vs average rating (scatter plot)

movie_stats = filtered_df.groupby(["movieId", "movie_title"]).agg(
    rating_count=("user_rating", "count"),
    movies_avg_rating=("movies_avg_rating", "first")  
).reset_index()

fig_scatter = px.scatter(
    movie_stats,
    x="rating_count",
    y="movies_avg_rating",  
    hover_name="movie_title",
    labels={
        "rating_count": "Number of Ratings",
        "movies_avg_rating": "Average Rating"
    },
    template="plotly_white"
)

fig_scatter.update_traces(marker=dict(size=5, color='#5ce1e6', opacity=0.6))
fig_scatter.update_layout(
        shapes=[  
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color='rgba(160,160,160,0.5)', width=1.5),
                layer='below'
            )
        ],
        plot_bgcolor='rgba(255,255,255,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        xaxis_tickangle=-45,
        showlegend=False,
        margin=dict(l=10, r=10, t=0, b=0),
        height=300,  
    )

st.plotly_chart(fig_scatter, use_container_width=True)


