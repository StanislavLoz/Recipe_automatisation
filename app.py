import re
import openai

import time
import math
from flask import Flask, request, render_template, session, make_response, render_template_string
import pandas as pd
from flask_session import Session
from flask import send_file
import os
from werkzeug.utils import secure_filename

# OpenAI API Key
openai.api_key = 'open_ai_key'


def ask_gpt3(prompt, temperature, max_tokens):
    """Use the GPT model to generate a response based on a given prompt"""
    # Set the maximum number of retries
    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a creative and intelligent cooking assistant. Follow user's requirements caarefully and to the letter. Minimize any other prose"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(1)  # Wait for 1 second before retrying

    # If we've retried max_retries times and still haven't succeeded, raise an error
    raise Exception("Exceeded maximum retries due to API overload.")

def generate_document_code(recipe_name):
    """Generate the document code for the recipe"""
    document_code = 'recette_' + recipe_name.lower().replace(' ', '_')
    return document_code


def generate_recette_texte_de_la_recette(steps):
    """Generate content for the recette_texte_de_la_recette column"""
    html_steps = []
    for i, step in enumerate(steps):
        html_step = f'<p><strong>Étape {i+1}:</strong> {step}</p>'
        html_steps.append(html_step)
    return ''.join(html_steps)

def generate_recette_handle(keywords):
    """Generate a unique handle for the recipe"""
    file_path = session.get('file_path', None)  # Retrieve file path from session
    if not file_path:
        raise ValueError("No file path found in session.")

    # Load the existing data
    df = pd.read_csv(file_path)

    # Define the initial prompt
    prompt = f"Given these keywords {keywords}, generate a unique URL handle. The handle should be all lowercase, You CAN'T use any specific char as accent, punctuation. The words within the string must be separated by a -, example : recette-pintade-a-la-biere. Respond in following format, Handle: recette-pintade-a-la-biere"

    while True:
        # Ask GPT-4 to generate a handle
        new_handle = ask_gpt3(prompt, temperature=0.7, max_tokens=50)
        print(new_handle)
        new_handle = re.findall('(?<=Handle:)(.*?)(?=\n|$)', new_handle, re.IGNORECASE)[0].strip()
        # Create a handle by replacing spaces with hyphens and converting to lowercase
        new_handle = new_handle.lower().replace(' ', '-')

        # If the handle is not in the CSV, break the loop
        if new_handle not in df['recette_handle'].values:
            break
        else:
            # Update the prompt to include the handle that needs to be avoided
            prompt = f"Given these keywords {keywords}, generate a unique URL handle. The handle should be all lowercase, You CAN'T use any specific char as accent, punctuation. The words within the string must be separated by a -, example : recette-pintade-a-la-biere. The handle should not be: {new_handle}. Respond only with the handle."

    return new_handle


def process_oven_time(content):
    # Extracting the time information
    oven_time = content.strip()
    #print(oven_time)
    # Check if oven_time is empty or '0'
    numeric_component = re.search(r'\d+', oven_time)
    if not oven_time or (numeric_component and int(numeric_component.group()) == 0):
        return None

    # Initialize variables to store hours and minutes
    hours = 0
    minutes = 0

    # Check if 'hour' is in the time
    if 'hour' in oven_time:
        hours_str = re.search(r'(\d+)\s*hour', oven_time)
        if hours_str:
            hours = int(hours_str.group(1))

    # Check if 'minute' is in the time
    if 'minute' in oven_time:
        minutes_str = re.search(r'(\d+)\s*minute', oven_time)
        if minutes_str:
            minutes = math.ceil(int(minutes_str.group(1)) / 5) * 5  # Round minutes to the nearest 5

    # If there's no 'hour' or 'minute' in the time, assume the number is minutes
    if 'hour' not in oven_time and 'minute' not in oven_time:
        minutes = math.ceil(int(oven_time) / 5) * 5  # Round minutes to the nearest 5
        print(minutes)

    # Construct the final output
    output = []
    if hours > 0:
        output.append(f'{hours} hour' + ('s' if hours > 1 else ''))
    if minutes > 0:
        output.append(f'{minutes} minutes')

    #print(output)

    return ' '.join(output)


def process_preparation_time(content):
    # Extracting the time information
    oven_time = content.strip()

    # Check if oven_time is empty or '0'
    if not oven_time or oven_time == '0':
        return None

    # Initialize variables to store hours and minutes
    hours = 0
    minutes = 0

    # Check if 'hour' is in the time
    if 'hour' in oven_time:
        hours_str = re.search(r'(\d+)\s*hour', oven_time)
        if hours_str:
            hours = int(hours_str.group(1))

    # Check if 'minute' is in the time
    if 'minute' in oven_time:
        minutes_str = re.search(r'(\d+)\s*minute', oven_time)
        if minutes_str:
            minutes = math.ceil(int(minutes_str.group(1)) / 5) * 5  # Round minutes to the nearest 5

    # Construct the final output
    output = []
    if hours > 0:
        output.append(f'{hours}_hour' + ('s' if hours > 1 else ''))
    if minutes > 0:
        output.append(f'{minutes}_minutes')

    return '_'.join(output)


def remove_step_numbers(step_text):
    # Removes everything before and including ":"
    step_text = re.sub(r'.*?: ', '', step_text)
    # Splits the steps into a list
    step_list = step_text.split('\n')
    # Removes empty strings in list
    step_list = [step for step in step_list if step]
    return step_list

def format_serving_size(serving_size_text):
    # List of allowed serving sizes
    allowed_sizes = [2, 4, 6, 8, 10, 12]

    # Extracts the first number in the string
    serving_number = int(re.search(r'\d+', serving_size_text).group())

    # Finds the smallest allowed serving size that is greater than or equal to serving_number
    closest_allowed_size = next((x for x in allowed_sizes if x >= serving_number), allowed_sizes[-1])

    # Formats the serving size
    formatted_serving_size = f"{closest_allowed_size}_personnes"

    return formatted_serving_size


def extract_temperature(content):
    # Try to extract temperature in Celsius
    match = re.search(r'(\d+)(?=°C)', content, re.IGNORECASE)
    if match:
        if match.group() == '0' or match.group() == '0°C':
            return None
        return match.group() + "°C"

    # If no Celsius temperature, try to extract temperature in Fahrenheit and convert to Celsius
    match = re.search(r'(\d+)(?=°F)', content, re.IGNORECASE)
    if match:
        fahrenheit = int(match.group())
        celsius = round((fahrenheit - 32) * 5/9)
        return str(celsius) + "°C"

    # If no temperature at all, return None
    return None


def generate_meta_short_description(short_description):

    # Trim the description to end at the closest sentence to 180 characters.
    if len(short_description) > 180:
        closest_period_index = short_description[:180].rfind('.')
        if closest_period_index != -1:
            output = short_description[:closest_period_index + 1]
        else:
            output = short_description[:180]
    else:
        output = short_description

    return output

def ask_gpt(prompt):
    """Use the GPT model to generate a response based on a given prompt"""#
    # Set the maximum number of retries
    max_retries = 100

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a creative and intelligent cooking assistant. Follow user's requirements caarefully and to the letter. Minimize any other prose"},
                    {"role": "user", "content": f"Create a recipe in french using these key words: {prompt} You should respond strictly with Name of the dish, ingredients, steps, needed time in oven (if oven is not needed respond with 0), needed time to prepare the dish except the oven time, temperature for the oven (if the oven is not needed respond with 0). Follow this format of responce:\nName:\nLe Hachi Parmentier traditionnel\nIngridiences: \n- 600 g de pommes de terre\n- 600 g de bœuf haché\n- 1 oignon\n- 2 tomates\n- 2 jaunes d'œuf\n- 15 cl de crème fraîche\n- 100 g de gruyère râpé\n- 1 gousse d'ail\n- Sel et poivre\n- Persil\n- Muscade\n- 30 g de beurre\nSteps:\nEtape 1 : Epluchez les pommes de terre et faites-les cuire dans de l'eau ou au Cookeo pendant 14 minutes en morceaux.\n\nEtape 2 : Emincez l'oignon, réduisez l'ail en purée et coupez les tomates.\n\nEtape 3 : Faites revenir l'oignon dans un peu d'huile d'olive, ajoutez ensuite la viande hachée et l'ail en purée. Une fois que la viande commence a bien cuire, ajoutez les tomates, le sel, le poivre, le persil.\n\nEtape 4 : Quand les pommes de terre sont cuites, égouttez-les et réduisez les en purée à l'aide d'un presse-purée avec le beurre, la crème, les jaunes d'œuf, un peu de sel et de muscade pour donner du goût à la purée. Poivrez la purée également et ajoutez-y le gruyère râpé.\n\nEtape 5 :  Dans un plat à four rectangulaire, placez la viande hachée et répartissez. Etalez ensuite la purée au-dessus et parsemez de fromage râpé pour gratiner.\n\nEtape 6 : Enfournez pendant 20 minutes à 210°C.\nOven time: 25 minutes\nPreparation time: 15_minutes\nTemperature: 180°C\nDocument code: based on you recepite title with underscore for words, example : recette_pintade_a_la_biere\nClassification code: classification_cuisine (a choice between classification_cuisine, classification_patisserie, classification_boulangerie, classification_boisson)\nShort description: beautifull and long description of the recipe, max 450 chars\nTags: cuisine|soupe|sans_gluten (tags from this list 'tous', 'autre', 'boisson', 'boulangerie', 'cocktail', 'confiture', 'cuisine', 'cuisine_americaine', 'cuisine_asiatique', 'cuisine_du_monde', 'cuisine_espagnole', 'cuisine_italienne', 'cuisine_latine', 'cuisine_traditionnelle', 'dessert', 'gâteau', 'noël', 'patisserie', 'post', 'sans_glutenk', 'soupe)\nDifficulty: facile (one choice from facile, tres_facile, difficile, moyen)\nBudget: (one choice from tres_economique, economique, cout_moyen, assez_cher, cher)\nAdvise: A small advise for cooking the recipe\nServing size: 4_personnes (one choice from 2_personnes, 4_personnes, 6_personnes, 8_personnes, 10_personnes, 12_personnes)."}
                ],
                max_tokens=3000,
                temperature = 0.5
            )
            content = response['choices'][0]['message']['content']
            print(content)
            name = re.findall('(?<=Name:)[\n ]*(.*?)(?=\n|$)', content, re.IGNORECASE | re.DOTALL)
            name += re.findall('(?<=Nom:)[\n ]*(.*?)(?=\n|$)', content, re.IGNORECASE | re.DOTALL)
            name = name[0].strip()
            print("name:", name)

            document_code = generate_document_code(name)
            print(document_code)

            matches = re.findall('(?<=Ingredients:\n)(.*?)(?=\n|$)(?=\nSteps)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingredients:\n)(.*?)(?=\nÉtapes)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingredients:\n)(.*?)(?=\nEtapes)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingridiences:\n)(.*?)(?=\n|$)(?=\nSteps)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingridiences:\n)(.*?)(?=\n|$)(?=\nÉtapes)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingridiences:\n)(.*?)(?=\n|$)(?=\nEtapes)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingrédients:\n)(.*?)(?=\n|$)(?=\nSteps)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingrédients:\n)(.*?)(?=\n|$)(?=\nÉtapes)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingrédients:\n)(.*?)(?=\n|$)(?=\nEtapes)', content, re.IGNORECASE | re.DOTALL)

            ingredients = matches[0].strip()
            print("ingredients:", ingredients)

            steps = re.findall('(?<=Steps:\n)(.*?)(?=\n|$)(?=\nOven time)', content, re.IGNORECASE | re.DOTALL)
            steps += re.findall('(?<=Étapes:\n)(.*?)(?=\n|$)(?=\nOven time)', content, re.IGNORECASE | re.DOTALL)
            steps += re.findall('(?<=Etapes:\n)(.*?)(?=\n|$)(?=\nOven time)', content, re.IGNORECASE | re.DOTALL)
            steps += re.findall('(?<=Steps:\n)(.*?)(?=\n|$)(?=\nTemps de cuisson au four)', content,
                                re.IGNORECASE | re.DOTALL)
            steps += re.findall('(?<=Étapes:\n)(.*?)(?=\n|$)(?=\nTemps de cuisson au four)', content,
                                re.IGNORECASE | re.DOTALL)
            steps += re.findall('(?<=Etapes:\n)(.*?)(?=\n|$)(?=\nTemps de cuisson au four)', content,
                                re.IGNORECASE | re.DOTALL)
            steps = steps[0].strip()
            step_red = remove_step_numbers(steps)
            step_red = generate_recette_texte_de_la_recette(step_red)
            print("steps:", step_red)

            oven_time = re.findall('(?<=Oven time:)(.*?)(?=\n|$)', content, re.IGNORECASE)
            oven_time += re.findall('(?<=Temps de cuisson au four:)(.*?)(?=\n|$)', content, re.IGNORECASE)
            oven_time = oven_time[0].strip()

            oven_time = process_oven_time(oven_time)
            print("oven_time:", oven_time)

            preparation_time = re.findall('(?<=Preparation time: ).*', content, re.IGNORECASE)
            preparation_time += re.findall('(?<=Temps de préparation: ).*', content, re.IGNORECASE)

            preparation_time = preparation_time[0].strip()
            preparation_time = process_preparation_time(preparation_time)
            print("preparation_time:", preparation_time)

            temperature = re.findall('(?<=Temperature: ).*', content, re.IGNORECASE)
            temperature += re.findall('(?<=Température: ).*', content, re.IGNORECASE)
            temperature += re.findall('(?<=Température du four: ).*', content, re.IGNORECASE)
            temperature = temperature[0].strip()

            temperature = extract_temperature(temperature)
            print("temperature:", temperature)

            classification_code = \
            re.findall('(?<=Classification code: )(.*?)(?=\n|$)', content, re.IGNORECASE | re.MULTILINE)[0].strip().replace(" ", "_")
            if not classification_code.startswith("classification_"):
                classification_code = "classification_" + classification_code
            print("classification_code:", classification_code)

            short_description = re.findall('(?<=Short description: ).*', content, re.IGNORECASE)[0]
            print("short_description:", short_description)

            tags = re.findall('(?<=Tags: ).*', content, re.IGNORECASE)[0]
            print("tags:", tags)

            difficulty = re.findall('(?<=Difficulty: ).*', content, re.IGNORECASE)
            difficulty += re.findall('(?<=Difficulté: ).*', content, re.IGNORECASE)
            difficulty = difficulty[0].strip()

            print("difficulty:", difficulty)

            budget = re.findall('(?<=Budget: ).*', content, re.IGNORECASE)[0]
            print("budget:", budget)

            advise = re.findall('(?<=Advise: |Advice: ).*', content, re.IGNORECASE)
            advise += re.findall('(?<=Conseils: ).*', content, re.IGNORECASE)
            advise += re.findall('(?<=Conseil: ).*', content, re.IGNORECASE)
            advise = advise[0].strip()

            print("advise:", advise)

            serving_size = re.findall('(?<=Serving size: ).*', content, re.IGNORECASE)
            serving_size += re.findall('(?<=Portions: ).*', content, re.IGNORECASE)
            serving_size += re.findall('(?<=Portion: ).*', content, re.IGNORECASE)
            serving_size += re.findall('(?<=Taille de la portion: ).*', content, re.IGNORECASE)
            serving_size += re.findall('(?<=Taille des portions: ).*', content, re.IGNORECASE)
            serving_size += re.findall('(?<=Nombre de portions: ).*', content, re.IGNORECASE)
            serving_size = serving_size[0].strip()

            serving_size = serving_size.replace(" ", "_").rstrip(".")
            serving_size = format_serving_size(serving_size)
            print("serving_size:", serving_size)

            handle = generate_recette_handle(prompt)
            print("handle:", handle)

            short_meta_description = generate_meta_short_description(short_description)
            print("short_meta_description:", short_meta_description)

            # Now you can use these variables as needed
            return {
                'classification_code': classification_code,
                'document_code': document_code,
                'attribute_set_code': 'attr_set_recette',
                'recettes_name (fr_FR)': name,
                'recette_description_courte_de_la_recette_ (fr_FR)': short_description,
                'recette_texte_de_la_recette (fr_FR)': step_red,
                'recette_commande_de_recette_excelify': 'MERGE',
                'recette_commande_des_tags_excelify': 'REPLACE',
                'recette_publication': 'TRUE',
                'recette_suffixe_du_template': 'article-recette',
                'recette_nom_de_l_auteur': 'romane_soppelsa',
                'recette_handle': handle,
                'recette_tags': tags,
                'recette_temps_de_cuisson (fr_FR)': oven_time,
                'recette_difficulte': difficulty,
                'recette_budget': budget,
                'recette_la_petite_astuce_fackelmann (fr_FR)': advise,
                'recette_tous_les_ingredients (fr_FR)': ingredients,
                'recette_nombre_de_personnes': serving_size,
                'recette_temps_de_preparation': preparation_time,
                'recette_titre_seo_de_la_recette (fr_FR)': name,

                'recette_meta_description_seo (fr_FR)': short_meta_description,
                'temperature_de_cuisson': temperature
            }

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(1)  # Wait for 1 second before retrying
            continue


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'supersecretkey'
Session(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create the folder if it doesn't exist

@app.route("/clear_stored_recipes", methods=["POST"])
def clear_stored_recipes():
    session.pop('stored_recipes', None)  # Remove the stored_recipes key from the session
    print('cleared')
    return "Stored recipes cleared."

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['file_path'] = file_path  # Save file path to session
        return 'File uploaded'
    else:
        return 'No file uploaded'

@app.route("/save", methods=["GET", "POST"])
def save():
    df = pd.DataFrame(session.get("stored_recipes", []))  # Get stored recipes from the session

    if not df.empty:
        filename = 'recipes.xlsx'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df.to_excel(file_path, index=False)

        session["stored_recipes"] = []

        response = make_response(send_file(file_path, as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
    else:
        return "No data to save."


@app.route("/regenerate", methods=["POST"])
def regenerate():
    if session["keywords"]:
        session["keyword_queue"] = list(session["keywords"])
        return process_next_keyword()
    else:
        return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        keywords = [keyword.strip() for keyword in request.form.get("keywords").split(',')]
        session["keywords"] = keywords
        session["keyword_queue"] = list(keywords)
        session["metadata_list"] = []
        if "stored_recipes" not in session:
            session["stored_recipes"] = []
        return process_next_keyword()
    return render_template("index.html")

@app.route("/process_next_keyword", methods=["GET", "POST"])
def process_next_keyword():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "Accept":
            session["stored_recipes"].append(session["metadata_list"][-1])
            return 'Recipe accepted!'

    if not session.get("keyword_queue") or not session["keyword_queue"]:
        return 'No more keywords left.'

    current_keyword = session["keyword_queue"].pop(0)
    metadata = ask_gpt(current_keyword)
    session["metadata_list"].append(metadata)

    return render_template_string(generate_recipe_html(current_keyword, metadata))

def generate_recipe_html(keyword, metadata):
    recipe_html = f'<h1>Generated Recipe for "{keyword}"</h1><pre>'
    for key, value in metadata.items():
        recipe_html += f'<span class="key">{key}:</span> <span class="value">{value}</span><br>'
    recipe_html += '</pre>'
    return recipe_html

if __name__ == "__main__":
    app.run(debug=True)