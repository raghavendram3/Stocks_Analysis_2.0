mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
address = \"0.0.0.0\"\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = \"#1E88E5\"\n\
backgroundColor = \"#FFFFFF\"\n\
secondaryBackgroundColor = \"#F5F5F5\"\n\
textColor = \"#212121\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml