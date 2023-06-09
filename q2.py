import re
import openai
import pandas as pd
import time
import math
import tkinter as tk
from tkinter import messagebox, scrolledtext
import queue
from tkinter.font import Font

# OpenAI API Key
openai.api_key = 'your openai key'


def ask_gpt3(prompt, temperature, max_tokens):
    """Use the GPT model to generate a response based on a given prompt"""
    # Set the maximum number of retries
    max_retries = 10

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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
    # Load the existing data
    df = pd.read_csv('C:\\Fackelmann_Internship\\Automatisation_recepies\\recipe.csv')

    # Define the initial prompt
    prompt = f"Given these keywords {keywords}, generate a unique URL handle. The handle should be all lowercase, You CAN'T use any specific char as accent, punctuation. The words within the string must be separated by a -, example : recette-pintade-a-la-biere. Respond only with the handle."

    while True:
        # Ask GPT-4 to generate a handle
        new_handle = ask_gpt3(prompt, temperature=0.7, max_tokens=50)

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
    # Removes "Step 1:", "Step 2:", etc.
    step_text = re.sub(r'Step \d+: ', '', step_text, flags=re.IGNORECASE)
    # Removes "1.", "2.", etc.
    step_text = re.sub(r'\d+\. ', '', step_text)
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
    """Use the GPT model to generate a response based on a given prompt"""
    # Set the maximum number of retries
    max_retries = 100

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative and intelligent cooking assistant. Follow user's requirements caarefully and to the letter. Minimize any other prose"},
                    {"role": "user", "content": f"Create a recipe using these key words: {prompt} You should respond strictly with Name of the dish, ingredients, steps, needed time in oven (if oven is not needed respond with 0), needed time to prepare the dish except the oven time, temperature for the oven (if the oven is not needed respond with 0). Follow this format of responce:\nName:\nLe Hachi Parmentier traditionnel\nIngridiences: \n- 600 g de pommes de terre\n- 600 g de bœuf haché\n- 1 oignon\n- 2 tomates\n- 2 jaunes d'œuf\n- 15 cl de crème fraîche\n- 100 g de gruyère râpé\n- 1 gousse d'ail\n- Sel et poivre\n- Persil\n- Muscade\n- 30 g de beurre\nSteps:\nEtape 1 : Epluchez les pommes de terre et faites-les cuire dans de l'eau ou au Cookeo pendant 14 minutes en morceaux.\n\nEtape 2 : Emincez l'oignon, réduisez l'ail en purée et coupez les tomates.\n\nEtape 3 : Faites revenir l'oignon dans un peu d'huile d'olive, ajoutez ensuite la viande hachée et l'ail en purée. Une fois que la viande commence a bien cuire, ajoutez les tomates, le sel, le poivre, le persil.\n\nEtape 4 : Quand les pommes de terre sont cuites, égouttez-les et réduisez les en purée à l'aide d'un presse-purée avec le beurre, la crème, les jaunes d'œuf, un peu de sel et de muscade pour donner du goût à la purée. Poivrez la purée également et ajoutez-y le gruyère râpé.\n\nEtape 5 :  Dans un plat à four rectangulaire, placez la viande hachée et répartissez. Etalez ensuite la purée au-dessus et parsemez de fromage râpé pour gratiner.\n\nEtape 6 : Enfournez pendant 20 minutes à 210°C.\nOven time: 25 minutes\nPreparation time: 15_minutes\nTemperature: 180°C.\nDocument code: based on you recepite title with underscore for words, example : recette_pintade_a_la_biere\nClassification code: classification_cuisine (a choice between classification_cuisine, classification_patisserie, classification_boulangerie, classification_boisson)\nShort description: beautifull and long description of the recipe, max 450 chars\nTags: cuisine|soupe|sans_gluten (tags from this list 'tous', 'autre', 'boisson', 'boulangerie', 'cocktail', 'confiture', 'cuisine', 'cuisine_americaine', 'cuisine_asiatique', 'cuisine_du_monde', 'cuisine_espagnole', 'cuisine_italienne', 'cuisine_latine', 'cuisine_traditionnelle', 'dessert', 'gâteau', 'noël', 'patisserie', 'post', 'sans_glutenk', 'soupe)\nDifficulty: facile (one choice from facile, tres_facile, difficile, moyen)\nBudget: (one choice from tres_economique, economique, cout_moyen, assez_cher, cher)\nAdvise: A small advise for cooking the recipe\nServig size: 4_personnes (one choice from 2_personnes, 4_personnes, 6_personnes, 8_personnes, 10_personnes, 12_personnes)."}
                ],
                max_tokens=3000,
                temperature = 0.9
            )
            content = response['choices'][0]['message']['content']
            print(content)
            # Using regular expressions to extract the necessary information
            name = re.findall('(?<=Name:)[\n ]*(.*?)(?=\n|$)', content, re.IGNORECASE | re.DOTALL)[0].strip()
            print("name:", name)

            document_code = generate_document_code(name)
            print(document_code)

            matches = re.findall('(?<=Ingredients:\n)(.*?)(?=\nSteps|$)', content, re.IGNORECASE | re.DOTALL)
            matches += re.findall('(?<=Ingridiences:\n)(.*?)(?=\nSteps|$)', content, re.IGNORECASE | re.DOTALL)
            ingredients = matches[0].strip()
            print("ingredients:", ingredients)

            steps = re.findall('(?<=Steps:\n)(.*?)(?=\n|$)(?=\nOven time)', content, re.IGNORECASE | re.DOTALL)[0]
            step_red = remove_step_numbers(steps)
            step_red = generate_recette_texte_de_la_recette(step_red)
            print("steps:", step_red)

            oven_time = re.findall('(?<=Oven time:)(.*?)(?=\n|$)', content, re.IGNORECASE)[0]
            oven_time = process_oven_time(oven_time)
            print("oven_time:", oven_time)

            preparation_time = re.findall('(?<=Preparation time: ).*', content, re.IGNORECASE)[0]
            preparation_time = process_preparation_time(preparation_time)
            print("preparation_time:", preparation_time)

            temperature = re.findall('(?<=Temperature: ).*', content, re.IGNORECASE)[0]
            temperature = extract_temperature(temperature)
            print("temperature:", temperature)

            classification_code = re.findall('(?<=Classification code: )(.*?)(?=\n|$)', content, re.IGNORECASE | re.MULTILINE)[
                0].strip().replace(" ", "_")
            if not classification_code.startswith("classification_"):
                classification_code = "classification_" + classification_code
            print("classification_code:", classification_code)

            short_description = re.findall('(?<=Short description: ).*', content, re.IGNORECASE)[0]
            print("short_description:", short_description)

            tags = re.findall('(?<=Tags: ).*', content, re.IGNORECASE)[0]
            print("tags:", tags)

            difficulty = re.findall('(?<=Difficulty: ).*', content, re.IGNORECASE)[0]
            print("difficulty:", difficulty)

            budget = re.findall('(?<=Budget: ).*', content, re.IGNORECASE)[0]
            print("budget:", budget)

            advise = re.findall('(?<=Advise: |Advice: ).*', content, re.IGNORECASE)[0]
            print("advise:", advise)

            serving_size = re.findall('(?<=Serving size: ).*', content, re.IGNORECASE)[0]
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
                'recettes_name (fr_FR)': name,
                'short_description': short_description,
                'recette_texte_de_la_recette': step_red,
                'recette_commande_de_recette_excelify': 'MERGE',
                'recette_commande_des_tags_excelify': 'REPLACE',
                'recette_publication': 'TRUE',
                'recette_suffixe_du_template': 'article-recette',
                'recette_nom_de_l_auteur': 'romane_soppelsa',
                'recette_handle': handle,
                'tags': tags,
                'oven_time': oven_time,
                'difficulty': difficulty,
                'budget': budget,
                'advise': advise,
                'ingredients': ingredients,
                'serving_size': serving_size,
                'preparation_time': preparation_time,
                'recette_titre_seo_de_la_recette (fr_FR)': name,
                'steps': steps,
                'short_meta_description': short_meta_description,
                'temperature': temperature
            }

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(1)  # Wait for 1 second before retrying
            continue


# Create a queue to store all the keywords
keyword_queue = queue.Queue()

metadata_list = []
current_keyword = ""
metadata = {}


def regenerate():
    """Function to regenerate the recipe for the current keyword"""
    # Close the current window
    new_window.destroy()

    # Add the current keyword back to the queue
    keyword_queue.put(current_keyword)

    # Process the next keyword in the queue
    process_next_keyword()


def accept():
    """Function to accept the current recipe"""
    # Append the accepted metadata to the list
    metadata_list.append(metadata)

    # Destroy the new window
    new_window.destroy()

    # Process the next keyword in the queue
    process_next_keyword()


def process_next_keyword():
    """Function to process the next keyword in the queue"""
    if not keyword_queue.empty():
        global current_keyword
        current_keyword = keyword_queue.get()

        try:
            # Generate the metadata for the current keyword
            global metadata
            metadata = ask_gpt(current_keyword)

            # Create a new window to show the recipe
            global new_window
            new_window = tk.Toplevel(root)
            new_window.title("Generated Recipe")
            new_window.geometry('1200x800')
            new_window.configure(background='white')

            # Convert the metadata dictionary into a formatted string
            formatted_metadata = '\n\n'.join([f"{key}:\n{value}" for key, value in metadata.items()])

            # Create a text box to show the recipe
            text_box = scrolledtext.ScrolledText(new_window, width=80, height=20, bg='#f0f0f0', fg='black')
            text_box_font = Font(size=16, family='Helvetica')
            text_box.configure(font=text_box_font)
            text_box.insert('1.0', formatted_metadata)
            text_box.pack(padx=20, pady=20)

            # Create a frame for buttons
            button_frame = tk.Frame(new_window)
            button_frame.pack()

            # Create an "Accept" button to save the recipe
            accept_button = tk.Button(button_frame, text="Accept", command=accept, bg='#00b300', fg='white',
                                      font=('Helvetica', 14, 'bold'), padx=20, pady=10)
            accept_button.pack(side='left', padx=20)

            # Create a "Reject" button to discard the recipe
            reject_button = tk.Button(button_frame, text="Reject", command=new_window.destroy, bg='#ff0000', fg='white',
                                      font=('Helvetica', 14, 'bold'), padx=20, pady=10)
            reject_button.pack(side='left', padx=20)

            # Create a "Regenerate" button to regenerate the recipe
            regenerate_button = tk.Button(button_frame, text="Regenerate", command=regenerate, bg='#007fff', fg='white',
                                          font=('Helvetica', 14, 'bold'), padx=20, pady=10)
            regenerate_button.pack(side='left', padx=20)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            process_next_keyword()
    else:  # If the queue is empty, meaning all keywords have been processed
        # Convert the list of metadata dictionaries into a DataFrame
        df = pd.DataFrame(metadata_list)

        # Write the DataFrame to an Excel file
        df.to_excel(r'C:\Fackelmann_Internship\Automatisation_recepies\recipes.xlsx', index=False)

def submit():
    """Function that gets triggered when the submit button is clicked"""
    keywords = keywords_entry.get()
    if keywords:
        keywords = [keyword.strip() for keyword in keywords.split(',')]

        # Add all keywords to the queue
        for keyword in keywords:
            keyword_queue.put(keyword)

        # Process the next keyword in the queue
        process_next_keyword()



root = tk.Tk()
root.title("Recipe Generator")

# Create a label
keywords_label = tk.Label(root, text="Enter keywords separated by comma")
keywords_label.pack()

# Create a text entry
keywords_entry = tk.Entry(root, width=50)
keywords_entry.pack()

# Create a submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

root.mainloop()