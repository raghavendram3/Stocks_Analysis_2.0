import streamlit as st

# Configure page
st.set_page_config(
    page_title="My Website",
    page_icon="🏠",
    layout="wide"
)

# Header section
st.title("🏠 Welcome to My Website")
st.markdown("This is the main homepage for my website with multiple applications.")

# Main content
st.header("Available Applications")

# App cards in columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Stock Analysis Tool")
    st.write("Get financial data visualization and investment tips for stocks.")
    st.write("Enter any stock symbol to view interactive charts, key financial metrics, and basic investment analysis.")
    st.page_link("pages/1_Stock_Analysis.py", label="Go to Stock Analysis", icon="📈")

with col2:
    st.subheader("➕ More Apps Coming Soon")
    st.write("Additional applications will be added in the future.")
    
# Footer
st.markdown("---")
st.caption("© 2025 My Website. All rights reserved.")