import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


DATA_PATH = "../data"
FOODGRAM_FILE = "foodgram_aug_30_2023.csv"
DISEASE_INDEX_FILE = "disease_index_foodgram_aug_30_2023.csv"
FOOD_INDEX_FILE = "food_index_foodgram_aug_30_2023.csv"
TSNE_FILE = "tsne_of_foodgram_svd_aug_30_2023.csv"

FOODGRAM_PATH = os.path.join(DATA_PATH, FOODGRAM_FILE)
FOOD_INDEX_PATH = os.path.join(DATA_PATH, FOOD_INDEX_FILE)
DISEASE_INDEX_PATH = os.path.join(DATA_PATH, DISEASE_INDEX_FILE)
TSNE_PATH = os.path.join(DATA_PATH, TSNE_FILE)



foodgram_df = pd.read_csv(FOODGRAM_PATH)
food_index_df = pd.read_csv(FOOD_INDEX_PATH)
disease_index_df = pd.read_csv(DISEASE_INDEX_PATH)

food_names = list(food_index_df.name)
food_names.sort()
disease_names = list(disease_index_df.name)
disease_names.sort()

DIM = 31
DEFAULT_SELECTION = "None"
MIN_FOOD_COUNT = 10
MAX_FOOD_COUNT = len(food_names)
DEFAULT_FOOD_COUNT = 10
MIN_DISEASE_COUNT = 10
MAX_DISEASE_COUNT = len(disease_names)
DEFAULT_DISEASE_COUNT = 10
FIG_OPACITY = 0.1
FIG_WIDTH = 580
FIG_HEIGHT = 500
FIG_OPACITY = 1
FONT_SIZE = 23
MARKER_SIZE = 10

def main():
    st.markdown("<h1 style='text-align: center; color: black;'>FOODGRAM</h1>", unsafe_allow_html=True)
    st.sidebar.header("Make your selection")
    initial_selection = st.sidebar.radio("", ["None", "Explore FOODGRAM", "Explore embeddings of FOODGRAM"])
    if initial_selection == "Explore FOODGRAM":
        st.sidebar.header("What search do you want to start?")
        food_disease_selection = st.sidebar.radio("", ["Food", "Disease"])
        st.sidebar.header("Enter your search")
        if food_disease_selection == "Food":
            selected_food = get_search_term(food_names)  
            disease_count = st.sidebar.slider('Number of Diseases to return', MIN_DISEASE_COUNT, MAX_DISEASE_COUNT, DEFAULT_DISEASE_COUNT)
            if selected_food != "None":
                st.write(get_disease_table(selected_food, disease_count))
        else:
            selected_disease = get_search_term(disease_names)
            food_count = st.sidebar.slider('Number of Food to return', MIN_FOOD_COUNT, MAX_FOOD_COUNT, DEFAULT_FOOD_COUNT)
            if selected_disease != "None":
                st.write(get_food_table(selected_disease, food_count))
    elif initial_selection == "Explore embeddings of FOODGRAM":
        plot_tsne_embeddings()
        st.markdown("<h4 style='text-align: center; color: black;'>Explore embedding space?</h1>", unsafe_allow_html=True)
        explore_embedding = st.radio("", ["Yes", "No"], index=1)
        if explore_embedding == "Yes":            
            foodgram = foodgram_df.to_numpy()[:,2:].astype(float)
            food_embeddings, disease_embeddings = get_embeddings(foodgram)
            total_embeddings = np.concatenate([food_embeddings, disease_embeddings])
            st.markdown("<h4 style='text-align: center; color: black;'>Enter the name of an entity</h1>", unsafe_allow_html=True)
            selected_entity = st.selectbox(" ", [DEFAULT_SELECTION]+food_names+disease_names, index=0)
            if selected_entity != DEFAULT_SELECTION:    
                food_count_2 = st.slider('Number of Food to return from embedding space', MIN_FOOD_COUNT, MAX_FOOD_COUNT, DEFAULT_FOOD_COUNT)
                disease_count_2 = st.slider('Number of Diseases to return from embedding space', MIN_DISEASE_COUNT, MAX_DISEASE_COUNT, DEFAULT_DISEASE_COUNT)
                plot_tsne_embeddings_after_entity_selection(selected_entity)
                tsne_df = pd.read_csv(TSNE_PATH) 
                embedding_index = tsne_df[tsne_df.name == selected_entity].index.values[0]
                sel_embedding = total_embeddings[embedding_index]                
                dot_product_list = []
                for index, item in enumerate(total_embeddings):                    
                    dot_product_dict = {}
                    dot_product_dict["name"] = tsne_df.iloc[index]["name"]
                    dot_product_dict["nodetype"] = tsne_df.iloc[index]["nodetype"]
                    dot_product_dict["similarity score"] = np.dot(sel_embedding, item)
                    dot_product_list.append(dot_product_dict)
                dot_product_df = pd.DataFrame(dot_product_list)
                dot_product_df_food = dot_product_df[dot_product_df.nodetype == "Food"]
                dot_product_df_disease = dot_product_df[dot_product_df.nodetype == "Disease"]
                
                dot_product_df_food_top = dot_product_df_food.sort_values(by="similarity score", ascending=False).reset_index().drop("index", axis=1).head(food_count_2)                
                dot_product_df_disease_top = dot_product_df_disease.sort_values(by="similarity score", ascending=False).reset_index().drop("index", axis=1).head(disease_count_2)
                st.markdown("<h4 style='text-align: center; color: black;'>Food entities that have similiar embeddings to {}</h4>".format(selected_entity), unsafe_allow_html=True)
                st.write(dot_product_df_food_top)
                st.markdown("<h4 style='text-align: center; color: black;'>Disease entities that have similiar embeddings to {}</h4>".format(selected_entity), unsafe_allow_html=True)
                st.write(dot_product_df_disease_top)


def plot_tsne_embeddings_after_entity_selection(selected_entity):
    tsne_df = pd.read_csv(TSNE_PATH) 
    tsne_df.loc[tsne_df.name == selected_entity, "color"] = "darkred"
    tsne_df["color"].fillna("gray", inplace=True)
    color_discrete_map = {"gray": "gray", "darkred": "darkred"}
    fig = px.scatter(tsne_df, 
                     x="tsne1", y="tsne2", 
                     color="color",
                     color_discrete_map=color_discrete_map,
                     opacity=1, 
                     hover_name="name",
                     hover_data=["nodetype"]                     
                    )
    fig.update_traces(marker=dict(size=MARKER_SIZE))
    fig.update_traces(opacity=0.2, selector=dict(marker_color="gray"))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=40), 
        width=FIG_WIDTH+40,
        height=FIG_HEIGHT,
        showlegend=True,
        xaxis=dict(showgrid=False, showticklabels=False, titlefont=dict(size=FONT_SIZE), title=""),
        yaxis=dict(showgrid=False, showticklabels=False, titlefont=dict(size=FONT_SIZE), title=""),
        legend=dict(font=dict(size=FONT_SIZE-5)),
        legend_title_text='Entity Type',
        legend_title_font=dict(size=FONT_SIZE-5) 
    )
    fig.update_xaxes(zeroline=False, zerolinewidth=0, showline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, zerolinewidth=0, showline=False, showgrid=False)
    fig.update_traces(name="Others", selector=dict(marker_color="gray"))
    fig.update_traces(name="Selected entity", selector=dict(marker_color="darkred"))

    
    fig.add_shape(type="line", x0=tsne_df["tsne1"].min(), y0=tsne_df["tsne2"].min(),
                  x1=tsne_df["tsne1"].max(), y1=tsne_df["tsne2"].min(),
                  line=dict(color="black", width=2))
    
    fig.add_shape(type="line", x0=tsne_df["tsne1"].min(), y0=tsne_df["tsne2"].min(),
                  x1=tsne_df["tsne1"].min(), y1=tsne_df["tsne2"].max(),
                  line=dict(color="black", width=2))
    
    fig.add_annotation(text="tSNE 1", x=tsne_df["tsne1"].mean(), y=tsne_df["tsne2"].min()-4,
                       xref="x", yref="y",
                       showarrow=False, font=dict(size=FONT_SIZE))
    
    fig.add_annotation(text="tSNE 2", x=tsne_df["tsne1"].min() - 5, y=tsne_df["tsne2"].mean(),
                       xref="x", yref="y",
                       showarrow=False, font=dict(size=FONT_SIZE), textangle=-90)
    st.plotly_chart(fig)

    
def plot_tsne_embeddings():
    tsne_df = pd.read_csv(TSNE_PATH)
    color_discrete_map = {"Food": "lightblue", "Disease": "red"}
    fig = px.scatter(tsne_df, 
                     x="tsne1", y="tsne2", 
                     color="nodetype",
                     color_discrete_map=color_discrete_map,
                     opacity=FIG_OPACITY, 
                     hover_name="name",
                     hover_data=["nodetype"]                     
                    )
    fig.update_traces(marker=dict(size=MARKER_SIZE))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=40),
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        showlegend=True,
        xaxis=dict(showgrid=False, showticklabels=False, titlefont=dict(size=FONT_SIZE), title=""),
        yaxis=dict(showgrid=False, showticklabels=False, titlefont=dict(size=FONT_SIZE), title=""),
        legend=dict(font=dict(size=FONT_SIZE-5)),
        legend_title_text='Entity Type',
        legend_title_font=dict(size=FONT_SIZE-5) 
    )
    
    fig.update_xaxes(zeroline=False, zerolinewidth=0, showline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, zerolinewidth=0, showline=False, showgrid=False)
    
    fig.add_shape(type="line", x0=tsne_df["tsne1"].min(), y0=tsne_df["tsne2"].min(),
                  x1=tsne_df["tsne1"].max(), y1=tsne_df["tsne2"].min(),
                  line=dict(color="black", width=2))
    
    fig.add_shape(type="line", x0=tsne_df["tsne1"].min(), y0=tsne_df["tsne2"].min(),
                  x1=tsne_df["tsne1"].min(), y1=tsne_df["tsne2"].max(),
                  line=dict(color="black", width=2))
    
    fig.add_annotation(text="tSNE 1", x=tsne_df["tsne1"].mean(), y=tsne_df["tsne2"].min()-4,
                       xref="x", yref="y",
                       showarrow=False, font=dict(size=FONT_SIZE))
    
    fig.add_annotation(text="tSNE 2", x=tsne_df["tsne1"].min() - 5, y=tsne_df["tsne2"].mean(),
                       xref="x", yref="y",
                       showarrow=False, font=dict(size=FONT_SIZE), textangle=-90)
    
    
    st.markdown("<h4 style='text-align: center; color: black;'>FOODGRAM based embeddings of Food and Disease entities</h4>", unsafe_allow_html=True)
    st.plotly_chart(fig)





        
            
    

    
    
def get_food_table(disease_name, food_count):
    diseaes_id = disease_index_df[disease_index_df.name==disease_name].identifier.values[0]    
    food_index_df_score = food_index_df.copy()
    food_index_df_score["FOODGRAM score"] = foodgram_df[diseaes_id]
    food_index_df_score = food_index_df_score.rename(columns={"name":"food name"})
    return food_index_df_score.sort_values(by="FOODGRAM score", ascending=False).reset_index().drop("index", axis=1).head(food_count)[["food name", "FOODGRAM score"]]
    
    
def get_disease_table(food_name, disease_count):
    disease_index_df_score = disease_index_df.copy()
    food_sel = food_index_df[food_index_df.name == food_name]["identifier"].values[0]
    disease_index_df_score["FOODGRAM score"] = foodgram_df[foodgram_df.identifier==food_sel].values[0][2:]
    disease_index_df_score = disease_index_df_score.rename(columns={"name":"disease name"})
    return disease_index_df_score.sort_values(by="FOODGRAM score", ascending=False).reset_index().drop("index", axis=1).head(disease_count)[["disease name", "FOODGRAM score"]]
    
def get_embeddings(foodgram_mat):
    U, S, Vh = np.linalg.svd(foodgram_mat, full_matrices=True, compute_uv=True, hermitian=False)
    return U[:, :DIM], Vh[:DIM, :].transpose()

def get_search_term(name_list):
    return st.sidebar.selectbox(" ", [DEFAULT_SELECTION]+name_list, index=0)


if __name__ == "__main__":
    main()