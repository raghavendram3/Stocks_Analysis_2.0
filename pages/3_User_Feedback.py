import streamlit as st
import json
import os
import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="StockTrackPro - User Feedback",
    page_icon="📝",
    layout="wide"
)

# Create data directory if it doesn't exist
data_dir = Path("./data")
feedback_file = data_dir / "feedback.json"

if not data_dir.exists():
    data_dir.mkdir(parents=True)
    
if not feedback_file.exists():
    with open(feedback_file, "w") as f:
        json.dump([], f)

# Function to load existing feedback
def load_feedback():
    try:
        with open(feedback_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

# Function to save feedback
def save_feedback(feedback_data):
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

# Header
st.title("📝 User Feedback")
st.markdown("Share your thoughts, suggestions, or report issues anonymously")

# Create tabs for submitting and viewing feedback
submit_tab, view_tab = st.tabs(["Submit Feedback", "View Feedback"])

with submit_tab:
    st.subheader("Submit Your Feedback")
    
    # Feedback form
    with st.form(key="feedback_form"):
        feedback_type = st.selectbox(
            "Feedback Type",
            options=["General Feedback", "Feature Request", "Bug Report", "Question"]
        )
        
        feedback_title = st.text_input("Subject (optional)")
        
        feedback_message = st.text_area(
            "Your Message",
            height=150,
            placeholder="Type your feedback here..."
        )
        
        rating = st.slider(
            "Rate Your Experience (1-5 stars)",
            min_value=1,
            max_value=5,
            value=4
        )
        
        submit_button = st.form_submit_button(label="Submit Feedback")
        
        if submit_button and feedback_message:
            # Load existing feedback
            all_feedback = load_feedback()
            
            # Add new feedback with timestamp
            new_feedback = {
                "type": feedback_type,
                "title": feedback_title if feedback_title else f"{feedback_type} on {datetime.datetime.now().strftime('%Y-%m-%d')}",
                "message": feedback_message,
                "rating": rating,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "id": len(all_feedback) + 1
            }
            
            all_feedback.append(new_feedback)
            
            # Save updated feedback
            save_feedback(all_feedback)
            
            st.success("Thank you for your feedback! It has been submitted successfully.")
            st.balloons()
        elif submit_button:
            st.error("Please enter a message before submitting.")

with view_tab:
    st.subheader("Recent Feedback")
    
    # Load and display feedback
    all_feedback = load_feedback()
    
    if not all_feedback:
        st.info("No feedback submitted yet.")
    else:
        # Filter options
        filter_container = st.container()
        col1, col2 = filter_container.columns(2)
        
        with col1:
            filter_type = st.multiselect(
                "Filter by Type",
                options=["All"] + list(set(item["type"] for item in all_feedback)),
                default=["All"]
            )
        
        with col2:
            sort_order = st.selectbox(
                "Sort by",
                options=["Newest First", "Oldest First", "Highest Rating", "Lowest Rating"]
            )
        
        # Apply filters
        filtered_feedback = all_feedback
        if filter_type and "All" not in filter_type:
            filtered_feedback = [item for item in all_feedback if item["type"] in filter_type]
        
        # Apply sorting
        if sort_order == "Newest First":
            filtered_feedback = sorted(filtered_feedback, key=lambda x: x["timestamp"], reverse=True)
        elif sort_order == "Oldest First":
            filtered_feedback = sorted(filtered_feedback, key=lambda x: x["timestamp"])
        elif sort_order == "Highest Rating":
            filtered_feedback = sorted(filtered_feedback, key=lambda x: x["rating"], reverse=True)
        elif sort_order == "Lowest Rating":
            filtered_feedback = sorted(filtered_feedback, key=lambda x: x["rating"])
        
        # Display feedback in expandable containers
        for item in filtered_feedback:
            with st.expander(f"{item['title']} ({item['type']}) - {'⭐' * item['rating']}"):
                st.markdown(f"**Submitted:** {item['timestamp']}")
                st.markdown(f"**Message:**\n{item['message']}")
                
                # Only show upvote/reply features if we implement user authentication later
                # For now, we'll keep it anonymous
        
        # Download option
        st.download_button(
            label="Download All Feedback (JSON)",
            data=json.dumps(all_feedback, indent=4),
            file_name="stocktrackpro_feedback.json",
            mime="application/json"
        )

# Add some information about privacy
st.markdown("---")
st.markdown("""
### Privacy Note
- Feedback is completely anonymous - we don't collect any personally identifiable information
- All feedback is stored locally and is used only to improve the application
- You can delete your feedback by accessing the feedback.json file directly if needed
""")