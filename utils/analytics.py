import streamlit as st
import streamlit.components.v1 as components

def inject_google_tag_manager():
    """Inject Google Tag Manager scripts into the Streamlit app"""
    # Google Tag Manager head script
    gtm_head = """
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    })(window,document,'script','dataLayer','GTM-MGMLFNKF');</script>
    <!-- End Google Tag Manager -->
    """
    
    # Google Tag Manager body script
    gtm_body = """
    <!-- Google Tag Manager (noscript) -->
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-MGMLFNKF"
    height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
    <!-- End Google Tag Manager (noscript) -->
    """
    
    # Inject scripts into the Streamlit app
    components.html(
        f"{gtm_head}{gtm_body}",
        height=0,
        width=0,
    )

def track_event(event_name, event_data=None):
    """Track custom events with Google Tag Manager
    
    Args:
        event_name: Name of the event to track
        event_data: Dictionary of additional data to track with the event
    """
    if event_data is None:
        event_data = {}
    
    # JavaScript code to push event to dataLayer
    js_code = f"""
    <script>
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({{
        'event': '{event_name}',
        {','.join([f"'{k}': '{v}'" for k, v in event_data.items()])}
    }});
    </script>
    """
    
    # Inject the JavaScript code
    components.html(js_code, height=0, width=0)