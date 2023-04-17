# from turtle import width
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
# import os
# import subprocess
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


    # st.title(f"Welcome to GitHub Project Recommendation System And User Analytics")
st.set_page_config(
    page_title="GitHub Project Recommendation System",
    page_icon="GitHub-icon.png",
)
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative&display=swap');
h1
{
  color:black;
  font-size:40px;
}
hr.new5 {
  border: 5px dashed red;
  border-radius: 60px;
}

</style>

    
                        <div >
                        
                          <h1 style="font-family: 'Cinzel Decorative', cursive;">Welcome to GitHub Project Recommendation System And User Analytics</h1>
                        </div>
                       
<hr class="new5">
                    
                        """,
                        height=300,
                      width=700,
                      )
# if selected == "Knowledge Based":
    
  
