# Recipe_automatisation

Flask-Based Recipe Generator Application
This is a simple Flask web application that generates recipes based on keywords provided by the user. It uses an OpenAI model for generating the recipes and offers an option to save them as an xlsx file.

Features
Keyword-based recipe generation
Recipe approval and storage
Clearing stored recipes
File upload feature
Saving stored recipes to an Excel file
Installation
To run this application locally, you need Python and the necessary libraries installed. Here's how you can set up:

bash
Copy code
# Clone this repository
git clone https://github.com/StanislavLoz/Recipe_automatisation.git

# Go into the repository
cd Recipe_automatisation

# Install necessary libraries
pip install flask pandas openai flask_session werkzeug

# Put your openai API key, replace the 'open_ai_key' to your actual key

openai.api_key = 'open_ai_key'

# Run the app
python app.py
Deployment
This application is not suitable to be run on a production server in its current state. If you want to run this application in a production setting, consider using a production-ready WSGI server like Gunicorn, uWSGI, or mod_wsgi if you're using Apache.
