from flask import Flask, render_template, redirect, request, abort, send_file, url_for, jsonify
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, nnls, linprog
from itertools import combinations
import os
import json
import io
import requests
import re
import csv
import tempfile
import uuid
from datetime import datetime
import json as json_lib

# export GOOGLE_APPLICATION_CREDENTIALS="food-ai-455507-e2a9c115814e.json"     
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "food-ai-455507-e2a9c115814e.json"))
if os.path.exists(json_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path

# Diagnostics toggle (optional): set env DIAG_MODE=1 to include server-side errors in API responses
DIAG_MODE = os.getenv("DIAG_MODE", "0") not in ["0", "false", "False", ""]

# Mesh generation mode and solution limits for memory/time-constrained environments (e.g., Render)
# MESH_MODE: 'all' (default) | 'first' (only first solution) | 'none' (disable STL generation)
MESH_MODE = os.getenv("MESH_MODE", "all").strip().lower()
# MAX_SOLUTIONS: cap how many solution options we compute/return
try:
    MAX_SOLUTIONS = max(1, int(os.getenv("MAX_SOLUTIONS", "2")))
except Exception:
    MAX_SOLUTIONS = 2

# Mesh storage backend: 'gcs' (default) to upload to Google Cloud Storage, or 'local' to keep files in /tmp and serve directly
MESH_STORAGE = os.getenv("MESH_STORAGE", "gcs").strip().lower()

def _user_records_path():
    return os.path.join(tempfile.gettempdir(), "user_records.json")

def _load_user_records():
    try:
        path = _user_records_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        print(f"[WARN] Failed to load user records: {e}")
    return {}

def _save_user_records(records):
    try:
        path = _user_records_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f)
    except Exception as e:
        print(f"[WARN] Failed to save user records: {e}")

def save_user_record(user_id, user_info, daily_nutrition=None, recommendation=None):
    """Upsert one user record and append an optional recommendation history item."""
    records = _load_user_records()
    now = datetime.utcnow().isoformat() + "Z"

    if not user_id:
        user_id = str(uuid.uuid4())

    existing = records.get(user_id, {
        'user_id': user_id,
        'created_at': now,
        'updated_at': now,
        'user_info': {},
        'history': []
    })

    existing['updated_at'] = now
    existing['user_info'] = user_info or existing.get('user_info', {})

    if daily_nutrition is not None or recommendation is not None:
        existing.setdefault('history', []).append({
            'timestamp': now,
            'daily_nutrition': daily_nutrition or {},
            'recommendation': recommendation or {}
        })

    records[user_id] = existing
    _save_user_records(records)
    return existing

def get_user_record(user_id):
    records = _load_user_records()
    return records.get(user_id)

def _manifest_path():
    import tempfile
    return os.path.join(tempfile.gettempdir(), "meshes_manifest.json")

def _load_manifest():
    try:
        path = _manifest_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load mesh manifest: {e}")
    return {}

def _save_manifest(manifest):
    try:
        path = _manifest_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
    except Exception as e:
        print(f"[WARN] Failed to save mesh manifest: {e}")
 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/uploads"
bucket_name = "food-ai"

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        # Make the blob publicly readable
        blob.make_public()
        return True
    except DefaultCredentialsError:
        return False
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False

    # print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Load nutrition data from CSV file instead of USDA API
# CSV columns: category_id, category_name, Density (g/ml), Calories (kcal/g), 
#              Protein (g/g), Carbohydrates (g/g), Fat (g/g), Reference (FDC ID)
csv_data = []
csv_loaded = False

def load_nutrition_csv():
    """Load the nutrition data from CSV file into memory"""
    global csv_data, csv_loaded
    try:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoodSAM", "food_full_data_revised.csv"))
        if os.path.exists(csv_path):
            csv_data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data.append(row)
            csv_loaded = True
            print(f"Loaded nutrition data from CSV: {len(csv_data)} food items")
            return True
        else:
            print(f"CSV file not found at {csv_path}")
            return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

# Simple cache to reduce repeated lookups
_search_cache = {}
_nutrition_cache = {}

WORD_NUMBER_MAP = {
    'a': 1,
    'an': 1,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'half': 0.5,
    'dozen': 12,
}


def normalize_food_name(name):
    """Normalize food names for better CSV matching (e.g., apples -> apple)."""
    cleaned = (name or '').strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if cleaned.endswith('es') and len(cleaned) > 3:
        cleaned = cleaned[:-2]
    elif cleaned.endswith('s') and len(cleaned) > 2:
        cleaned = cleaned[:-1]
    return cleaned


def parse_direct_macro_input(text):
    """Parse direct macro entry like '100g carb' or '+30 protein'."""
    match = re.match(r'^([+-]?\d+(?:\.\d+)?)\s*g?\s*(carb|carbon|carbohydrate|protein|fat)s?$', text.strip(), re.IGNORECASE)
    if not match:
        return None

    amount = float(match.group(1))
    macro_type = match.group(2).lower()

    nutrition = {'carbs': 0.0, 'protein': 0.0, 'fat': 0.0, 'calories': 0.0}
    if macro_type in ['carb', 'carbon', 'carbohydrate']:
        nutrition['carbs'] = amount
        macro_label = 'carbs'
    elif macro_type == 'protein':
        nutrition['protein'] = amount
        macro_label = 'protein'
    else:
        nutrition['fat'] = amount
        macro_label = 'fat'

    return {
        'food_name': f"direct {macro_label}",
        'quantity': amount,
        'unit': 'g',
        'carbs': round(nutrition['carbs'], 2),
        'protein': round(nutrition['protein'], 2),
        'fat': round(nutrition['fat'], 2),
        'calories': 0.0
    }

def parse_food_input(food_input):
    """
    Parse user input like "100g chicken breast", "1 medium apple", or "two eggs".
    Returns: (food_name, quantity, unit)
    """
    cleaned = food_input.strip()

    # Define valid measurement units
    valid_units = {
        'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms',
        'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds',
        'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters',
        'cup', 'cups', 'tbsp', 'tablespoon', 'tablespoons',
        'tsp', 'teaspoon', 'teaspoons', 'piece', 'pieces',
        'serving', 'servings', 'slice', 'slices'
    }

    # Pattern: number + optional unit + food name (e.g., "100g chicken breast")
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s+(.+)$', cleaned)
    if match:
        quantity = float(match.group(1))
        raw_unit = (match.group(2) or '').lower()
        remaining_text = match.group(3).strip()

        # Check if raw_unit is a valid measurement unit
        if raw_unit and raw_unit in valid_units:
            # It's a valid unit, so remaining_text is the food name
            food_name = remaining_text
            unit = raw_unit
        elif raw_unit:
            # Not a valid unit - treat it as part of food name
            food_name = f"{raw_unit} {remaining_text}"
            # Determine unit based on food type
            countable_keywords = ['egg', 'eggs', 'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'bread', 'breads', 'hamburger', 'hamburgers', 'burger', 'burgers', 'sandwich', 'sandwiches']
            unit = 'unit' if any(k in food_name.lower() for k in countable_keywords) else 'g'
        else:
            # No unit detected, determine based on food type
            food_name = remaining_text
            countable_keywords = ['egg', 'eggs', 'apple', 'apples', 'banana', 'bananas', 'orange', 'oranges', 'bread', 'breads', 'hamburger', 'hamburgers', 'burger', 'burgers', 'sandwich', 'sandwiches']
            unit = 'unit' if any(k in food_name.lower() for k in countable_keywords) else 'g'

        return food_name, quantity, unit

    # Pattern: word-number + food name (e.g., "two eggs", "a banana")
    word_match = re.match(r'^(?P<num_word>[a-zA-Z]+)\s+(?P<food>.+)$', cleaned.lower())
    if word_match:
        num_word = word_match.group('num_word')
        food_name = word_match.group('food').strip()
        if num_word in WORD_NUMBER_MAP:
            return food_name, float(WORD_NUMBER_MAP[num_word]), 'unit'

    # Fallback: treat as a single unit
    return cleaned, 1.0, 'unit'

def search_csv_food(food_name):
    """
    Search for food in the loaded CSV data.
    Returns a dictionary matching the expected format.
    Uses improved matching: exact > prefix > word boundary > no substring fallback.
    """
    if not csv_loaded or not csv_data:
        print("CSV data not loaded")
        return None
    
    normalized_query = normalize_food_name(food_name)
    cache_key = normalized_query
    if cache_key in _search_cache:
        print(f"[CACHE HIT] Using cached results for '{food_name}'")
        return _search_cache[cache_key]

    exact_matches = []
    prefix_matches = []
    word_boundary_matches = []
    
    for row in csv_data:
        category_name = row.get('category_name', '')
        normalized_category = normalize_food_name(category_name)
        if not normalized_category or normalized_category == 'background':
            continue
        
        if normalized_query == normalized_category:
            # Exact match (highest priority)
            exact_matches.append(row)
        elif normalized_query and normalized_category.startswith(normalized_query):
            # Prefix match (e.g., "apple" matches "apple pie") - but only if it's a word boundary
            # Check that the next character after query is a space or end of string
            next_pos = len(normalized_query)
            if next_pos >= len(normalized_category) or normalized_category[next_pos] == ' ':
                prefix_matches.append(row)
        elif normalized_query and normalized_query in normalized_category.split():
            # Word boundary match (e.g., "apple" is a complete word in "green apple")
            word_boundary_matches.append(row)

    # Use the best match category available
    matching_foods = exact_matches if exact_matches else (
        prefix_matches if prefix_matches else word_boundary_matches
    )
    
    if not matching_foods:
        print(f"No foods found for '{food_name}' in CSV data")
        return None
    
    # Convert CSV rows to match USDA API response format
    foods = []
    for food_row in matching_foods:
        foods.append({
            'fdcId': str(food_row.get('Reference (FDC ID)', food_row.get('category_id', ''))),
            'description': food_row.get('category_name', ''),
            'dataType': 'CSV',
            'foodCategory': {'description': food_row.get('category_name', '')},
            # Store our own data for later retrieval
            '_csv_data': food_row
        })
    
    response = {'foods': foods}
    
    # Cache the result
    _search_cache[cache_key] = response
    print(f"[CACHED] Stored results for '{food_name}' - Found {len(foods)} items")
    
    return response

def get_food_nutrition_csv(fdc_id, csv_food_row, quantity, unit):
    """
    Get nutrition info from CSV data.
    csv_food_row: The CSV row dictionary containing nutrition data
    quantity: amount user consumed
    unit: unit of measurement (g, cup, etc.)
    """
    print(f"\nCSV Nutrition Data for {csv_food_row.get('category_name', 'Unknown')}")
    
    food_name = csv_food_row.get('category_name', '')
    
    # Extract nutrition values from CSV (values are per gram)
    try:
        cal_str = str(csv_food_row.get('Calories (kcal/g)', '0') or '0').strip()
        calories_per_gram = float(cal_str) if cal_str else 0
        
        prot_str = str(csv_food_row.get('Protein (g/g)', '0') or '0').strip()
        protein_per_gram = float(prot_str) if prot_str else 0
        
        carbs_str = str(csv_food_row.get('Carbohydrates (g/g)', '0') or '0').strip()
        carbs_per_gram = float(carbs_str) if carbs_str else 0
        
        fat_str = str(csv_food_row.get('Fat (g/g)', '0') or '0').strip()
        fat_per_gram = float(fat_str) if fat_str else 0
        
        density_str = str(csv_food_row.get('Density (g/ml)', '1') or '1').strip()
        density = float(density_str) if density_str else 1
    except Exception as e:
        print(f"Error parsing nutrition values: {e}")
        return None
    
    print(f"Food: {food_name}")
    print(f"Nutrition per gram: Calories={calories_per_gram}, Protein={protein_per_gram}g, Carbs={carbs_per_gram}g, Fat={fat_per_gram}g")
    
    # Convert quantity to grams
    quantity_in_grams = quantity
    unit_lower = unit.lower()
    food_name_lower = food_name.lower()
    
    # Descriptive sizes with USDA-style defaults
    if unit_lower in ['small', 'sm']:
        if 'egg' in food_name_lower:
            quantity_in_grams = quantity * 50
        elif 'apple' in food_name_lower:
            quantity_in_grams = quantity * 149
        elif 'banana' in food_name_lower:
            quantity_in_grams = quantity * 101
        else:
            quantity_in_grams = quantity * 100
    
    elif unit_lower in ['medium', 'med', 'md']:
        if 'egg' in food_name_lower:
            quantity_in_grams = quantity * 60
        elif 'apple' in food_name_lower:
            quantity_in_grams = quantity * 182
        elif 'banana' in food_name_lower:
            quantity_in_grams = quantity * 118
        elif 'orange' in food_name_lower:
            quantity_in_grams = quantity * 131
        else:
            quantity_in_grams = quantity * 150
    
    elif unit_lower in ['large', 'lg', 'big']:
        if 'egg' in food_name_lower:
            quantity_in_grams = quantity * 70
        elif 'apple' in food_name_lower:
            quantity_in_grams = quantity * 223
        elif 'banana' in food_name_lower:
            quantity_in_grams = quantity * 136
        else:
            quantity_in_grams = quantity * 200
    
    # Volume units (using density if available)
    elif unit_lower in ['cup', 'cups']:
        quantity_in_grams = quantity * 240 * density
    elif unit_lower in ['tbsp', 'tablespoon', 'tablespoons']:
        quantity_in_grams = quantity * 15 * density
    elif unit_lower in ['tsp', 'teaspoon', 'teaspoons']:
        quantity_in_grams = quantity * 5 * density
    
    # Weight units
    elif unit_lower in ['oz', 'ounce', 'ounces']:
        quantity_in_grams = quantity * 28.35
    elif unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
        quantity_in_grams = quantity * 453.59
    elif unit_lower in ['g', 'gram', 'grams']:
        quantity_in_grams = quantity
    
    # Liquid volume (ml)
    elif unit_lower in ['ml', 'milliliter', 'milliliters']:
        quantity_in_grams = quantity * density
    
    # Countable items
    elif unit_lower in ['piece', 'pieces', 'item', 'items', 'unit', 'units', 'egg', 'eggs', 'slice', 'slices', 'toast', 'toasts']:
        default_piece_weight = 150
        if 'egg' in food_name_lower:
            default_piece_weight = 60
        elif 'banana' in food_name_lower:
            default_piece_weight = 118
        elif 'apple' in food_name_lower:
            default_piece_weight = 182
        elif 'orange' in food_name_lower:
            default_piece_weight = 131
        elif 'bread' in food_name_lower:
            default_piece_weight = 30  # 1 slice of bread ≈ 30g
        quantity_in_grams = quantity * default_piece_weight
    
    else:
        # Unknown unit - assume grams
        print(f"[WARNING] Unknown unit '{unit}' - treating quantity as grams")
        quantity_in_grams = quantity
    
    print(f"Input: {quantity}{unit} = {quantity_in_grams:.2f}g")
    
    # Calculate based on per-gram values
    nutrition = {
        'carbs': round(carbs_per_gram * quantity_in_grams, 2),
        'protein': round(protein_per_gram * quantity_in_grams, 2),
        'fat': round(fat_per_gram * quantity_in_grams, 2),
        'calories': round(calories_per_gram * quantity_in_grams, 2)
    }
    
    print(f"Final nutrition: {nutrition}")
    print("=" * 50 + "\n")
    
    return {
        'food_name': food_name,
        'carbs': nutrition['carbs'],
        'protein': nutrition['protein'],
        'fat': nutrition['fat'],
        'calories': nutrition['calories'],
        'quantity': quantity,
        'unit': unit,
        'serving_size': 1  # CSV data is per gram, so serving is 1g
    }

# USDA API fallback functions
# Prefer USDA_API_KEY; fallback to DATA_GOV_API_KEY; final fallback to DEMO_KEY.
USDA_API_KEY = os.getenv("USDA_API_KEY") or os.getenv("DATA_GOV_API_KEY", "DEMO_KEY")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# Doubao LLM API configuration for nutrition queries
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "e1051380-9eac-4253-bd06-cc4fb1fb53db")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_MODEL = "doubao-1-5-pro-32k-250115"

def query_doubao_nutrition(food_name, quantity, unit):
    """
    Query Doubao LLM API to get nutrition data for a food.
    Returns nutrition dict or None.
    """
    try:
        prompt = f"{quantity}{unit} {food_name} has how many carbs, protein and fat? answer in the format: carbs: xxg, protein: xxg, fat: xxg. and do not say anything else"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOUBAO_API_KEY}"
        }
        
        payload = {
            "model": DOUBAO_MODEL,
            "messages": [
                {"role": "system", "content": "你是一个营养学家助手，专门提供食物的营养信息。"},
                {"role": "user", "content": prompt}
            ]
        }
        
        print(f"[Doubao] Querying nutrition for: {prompt}")
        response = requests.post(DOUBAO_BASE_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the assistant's response
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {})
            content = message.get('content', '').strip()
            print(f"[Doubao] Response: {content}")
            
            # Parse the response for nutrition values
            # Expected format: "carbs: 25g, protein: 5g, fat: 1g"
            nutrition = {
                'carbs': 0.0,
                'protein': 0.0,
                'fat': 0.0,
                'calories': 0.0
            }
            
            # Extract values using regex
            carbs_match = re.search(r'carbs?\s*:\s*([\d.]+)', content, re.IGNORECASE)
            protein_match = re.search(r'protein\s*:\s*([\d.]+)', content, re.IGNORECASE)
            fat_match = re.search(r'fat\s*:\s*([\d.]+)', content, re.IGNORECASE)
            
            if carbs_match:
                nutrition['carbs'] = float(carbs_match.group(1))
            if protein_match:
                nutrition['protein'] = float(protein_match.group(1))
            if fat_match:
                nutrition['fat'] = float(fat_match.group(1))
            
            if _is_nutrition_meaningful({'carbs': nutrition['carbs'], 'protein': nutrition['protein'], 'fat': nutrition['fat']}):
                result = {
                    'food_name': food_name,
                    'carbs': round(nutrition['carbs'], 2),
                    'protein': round(nutrition['protein'], 2),
                    'fat': round(nutrition['fat'], 2),
                    'calories': nutrition['calories'],
                    'quantity': quantity,
                    'unit': unit,
                    'source': 'Doubao LLM'
                }
                print(f"[Doubao] Extracted nutrition: {result}")
                return result
            else:
                print(f"[Doubao] Response parsed but nutrition not meaningful: {nutrition}")
                return None
        else:
            print(f"[Doubao] No choices in response")
            return None
            
    except Exception as e:
        print(f"[Doubao] Error querying nutrition: {e}")
        return None

def search_usda_food(food_name):
    """
    Search for food in USDA FoodData Central via API.
    Returns a dictionary with search results or None if not found.
    """
    try:
        print(f"\nSearching USDA API for: {food_name}")
        url = f"{USDA_BASE_URL}/foods/search"
        merged = []
        seen_ids = set()

        query_candidates = [food_name]
        normalized = normalize_food_name(food_name)
        if normalized and normalized != food_name:
            query_candidates.append(normalized)

        # Two-pass strategy: broad searches with increasing pageSize
        # USDA API doesn't support dataType filtering in search params
        for q in query_candidates:
            for pass_num in range(2):
                page_size = 25 if pass_num == 0 else 50
                params = {
                    'query': q,
                    'pageSize': page_size,
                    'api_key': USDA_API_KEY
                }
                try:
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    foods = data.get('foods', [])
                    for item in foods:
                        fdc_id = str(item.get('fdcId', ''))
                        if not fdc_id or fdc_id in seen_ids:
                            continue
                        seen_ids.add(fdc_id)
                        merged.append(item)
                except Exception as pass_err:
                    print(f"[USDA] Pass {pass_num + 1} error for '{q}': {pass_err}")
                    continue

        if merged:
            print(f"[USDA] Found {len(merged)} merged results for '{food_name}'")
            return {'foods': merged}

        print(f"[USDA] No results found for '{food_name}'")
        return None
    except Exception as e:
        print(f"[USDA] Error searching for '{food_name}': {e}")
        return None

def _is_nutrition_meaningful(nutrition):
    if not nutrition:
        return False
    return any(float(nutrition.get(k, 0) or 0) > 0 for k in ['carbs', 'protein', 'fat', 'calories'])

def _extract_usda_nutrition_per_100g(nutrients):
    """Extract macros from USDA nutrients across different payload shapes."""
    nutrition_per_100g = {
        'calories': 0.0,
        'protein': 0.0,
        'carbs': 0.0,
        'fat': 0.0
    }

    # USDA nutrientNumber reference: 1008=Energy kcal, 1003=Protein, 1005=Carbs, 1004=Fat
    for nutrient in nutrients or []:
        nutrient_obj = nutrient.get('nutrient', {}) if isinstance(nutrient, dict) else {}

        nutrient_name = str(
            nutrient_obj.get('name')
            or nutrient.get('nutrientName')
            or ''
        ).strip().lower()

        nutrient_number = str(
            nutrient_obj.get('number')
            or nutrient.get('nutrientNumber')
            or ''
        ).strip()

        amount = nutrient.get('amount')
        if amount is None:
            amount = nutrient.get('value', 0)

        try:
            amount = float(amount or 0)
        except Exception:
            amount = 0.0

        if nutrient_number == '1008' or ('energy' in nutrient_name and 'kcal' in nutrient_name):
            nutrition_per_100g['calories'] = amount
        elif nutrient_number == '1003' or 'protein' in nutrient_name:
            nutrition_per_100g['protein'] = amount
        elif nutrient_number == '1005' or ('carbohydrate' in nutrient_name and 'fiber' not in nutrient_name):
            nutrition_per_100g['carbs'] = amount
        elif nutrient_number == '1004' or ('fat' in nutrient_name and ('total' in nutrient_name or 'lipid' in nutrient_name)):
            nutrition_per_100g['fat'] = amount

    return nutrition_per_100g

def get_food_nutrition_usda(fdc_id, quantity, unit):
    """
    Get nutrition info from USDA API.
    fdc_id: FDC ID from USDA food search
    quantity: amount user consumed
    unit: unit of measurement (g, cup, etc.)
    """
    try:
        print(f"\nFetching USDA nutrition data for FDC ID: {fdc_id}")
        url = f"{USDA_BASE_URL}/food/{fdc_id}"
        params = {'api_key': USDA_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        food_data = response.json()
        food_name = food_data.get('description', 'Unknown Food')
        nutrients = food_data.get('foodNutrients', [])
        
        # Extract key nutrients (per 100g serving from USDA)
        nutrition_per_100g = _extract_usda_nutrition_per_100g(nutrients)
        
        # Convert quantity to grams (similar logic as CSV)
        quantity_in_grams = quantity
        unit_lower = unit.lower()
        food_name_lower = food_name.lower()
        
        # Descriptive sizes with USDA-style defaults
        if unit_lower in ['small', 'sm']:
            quantity_in_grams = quantity * 100
        elif unit_lower in ['medium', 'med', 'md']:
            quantity_in_grams = quantity * 150
        elif unit_lower in ['large', 'lg', 'big']:
            quantity_in_grams = quantity * 200
        elif unit_lower in ['cup', 'cups']:
            quantity_in_grams = quantity * 240
        elif unit_lower in ['tbsp', 'tablespoon', 'tablespoons']:
            quantity_in_grams = quantity * 15
        elif unit_lower in ['tsp', 'teaspoon', 'teaspoons']:
            quantity_in_grams = quantity * 5
        elif unit_lower in ['oz', 'ounce', 'ounces']:
            quantity_in_grams = quantity * 28.35
        elif unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            quantity_in_grams = quantity * 453.59
        elif unit_lower in ['ml', 'milliliter', 'milliliters']:
            quantity_in_grams = quantity
        elif unit_lower in ['g', 'gram', 'grams']:
            quantity_in_grams = quantity
        elif unit_lower in ['piece', 'pieces', 'item', 'items', 'unit', 'units', 'egg', 'eggs', 'slice', 'slices', 'toast', 'toasts']:
            default_piece_weight = 150
            quantity_in_grams = quantity * default_piece_weight
        else:
            quantity_in_grams = quantity
        
        # Calculate nutrition (USDA provides per 100g, so scale accordingly)
        multiplier = quantity_in_grams / 100.0
        
        nutrition = {
            'carbs': round(nutrition_per_100g['carbs'] * multiplier, 2),
            'protein': round(nutrition_per_100g['protein'] * multiplier, 2),
            'fat': round(nutrition_per_100g['fat'] * multiplier, 2),
            'calories': round(nutrition_per_100g['calories'] * multiplier, 2)
        }
        
        print(f"USDA Nutrition: {nutrition} (from {quantity}{unit} = {quantity_in_grams:.2f}g)")
        
        result = {
            'food_name': food_name,
            'carbs': nutrition['carbs'],
            'protein': nutrition['protein'],
            'fat': nutrition['fat'],
            'calories': nutrition['calories'],
            'quantity': quantity,
            'unit': unit,
            'source': 'USDA API'
        }
        if not _is_nutrition_meaningful(result):
            print(f"[USDA] Nutrition data incomplete for FDC ID {fdc_id}, skipping this candidate")
            return None

        return result
    except Exception as e:
        print(f"[USDA] Error fetching nutrition: {e}")
        return None

def get_food_nutrition_with_fallback(food_name, quantity, unit):
    """
    Query Doubao LLM API for nutrition data.
    Returns tuple: (nutrition_dict_or_none, source_label)
    """
    nutrition = query_doubao_nutrition(food_name, quantity, unit)
    if nutrition:
        return nutrition, "Doubao LLM"
    return None, ""

# Load CSV data on app startup
with app.app_context():
    load_nutrition_csv()

@app.route('/')
def main():
    # Redirect base URL to chatbot page
    return redirect('/chatbot')

@app.route('/nutrition_calculation', methods=["GET", "POST"])
def nutrition_calculation():
    path = request.args.get('path')
    name = path.split("/")[3].split('.')[0]
    with open('./static/foodseg/' + name + '/' + name + "_nutrition.json") as f: nutrition_data = json.load(f)
    results = {
        'image_origin': path,
        'image_seglab': './static/foodseg/' + name + '/' + name + "_labeled_seg.png",
        'image_report': nutrition_data
    }

    # nutrition = {
    #     'carbohydrate': round(nutrition_data['carbs'], 2),
    #     'protein': round(nutrition_data['protein'], 2),
    #     'fat': round(nutrition_data['fat'], 2)
    # }

    if request.method == "POST":
        next = request.form["next"]
        if next: return redirect(url_for("data_collection", carbs=round(nutrition_data['carbs'], 2), protein=round(nutrition_data['protein'], 2), fat=round(nutrition_data['fat'], 2)))

    return render_template("nutrition-calculation.html", results=results)

def calculate_rmr(weight, height, age, sex):
    if sex == 0:
        rmr = (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5 
    else:
        rmr = (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161
    return rmr

def calculate_daily_calories(rmr, activity_level):
    if activity_level == 0:
        calories = rmr * 1.2
    elif activity_level == 1:
        calories = rmr * 1.375
    elif activity_level == 2:
        calories = rmr * 1.55
    else:
        calories = rmr * 1.725
    return calories

def constraint_func(x): 
    return x[0] + x[1]

# x, y [5, 10], z in [1, 3]
def calculate_cube_dimension(volume):
    x = y = 10
    z = 3

    for i in np.arange(5.0, 10.0, 1.0):
        if volume / (i * i) <= 3 and volume / (i * i) >= 1:
            z = volume / (i * i)
            x = y = i

    return x * 10.0, y * 10.0, z * 10.0

min_size = np.array([8.0, 8.0, 0.15])  # Minimum dimensions in cm
max_size = np.array([15.0, 13.0, 2.2])  # Maximum dimensions in cm
MAX_VOLUME = max_size[0] * max_size[1] * max_size[2] # Dimention is cm
TOLERANCE = 400  # allow feasible solutions even with moderate error

def calculate_cube_dimension(volume):
    # Define size limits in cm

    # Calculate minimum and maximum volume based on maximum dimensions
    min_volume = min_size[0] * min_size[1] * min_size[2]
    max_volume = max_size[0] * max_size[1] * max_size[2]

    # Check if the volume is valid
    if volume < min_volume: return min_size[0] * 10.0, min_size[1] * 10.0,  min_size[2] * 10.0
    if volume > max_volume: return max_size[0] * 10.0, max_size[1] * 10.0,  max_size[2] * 10.0  # Return zero if volume is invalid

    # Iterate through possible dimensions
    for x in np.arange(min_size[0], max_size[0] + 0.1, 0.1):  # Increment by 0.5 cm
        for y in np.arange(min_size[1], max_size[1] + 0.1, 0.1):  # Increment by 0.5 cm
            z = volume / (x * y)  # Calculate height based on volume
            # Check if height is within limits
            if min_size[2] <= z <= max_size[2]:
                return x * 10.0, y * 10.0, z * 10.0  # Return dimensions in mm

    return 0, 0, 0 # Return zero if no valid dimensions are found

def mesh_generation(name, weight, density, z_offset=0.0): #g/cm3, z_offset in mm
    x, y, z = calculate_cube_dimension(weight / density) # in mm
    # print(name, weight, density, weight / density, x, y, z)
    if (x == 0 or y == 0 or z == 0): return 0, 0, 0
    # print(x, y, z)
    # If mesh generation is disabled, just return dimensions without creating STL
    if MESH_MODE == 'none':
        return x, y, z

    # Lazy import to avoid loading numpy-stl unless needed
    try:
        from stl import mesh as stl_mesh
    except Exception as e:
        print(f"[WARN] Failed to import numpy-stl: {e}. Skipping STL generation.")
        return x, y, z

    # Center the box on the XY plane so all items share the same vertical axis.
    # z_offset shifts this item up to sit on top of the previous item.
    hx, hy = x / 2.0, y / 2.0
    z0, z1 = z_offset, z_offset + z
    vertices = np.array([
        [-hx, -hy, z0],
        [ hx, -hy, z0],
        [ hx,  hy, z0],
        [-hx,  hy, z0],
        [-hx, -hy, z1],
        [ hx, -hy, z1],
        [ hx,  hy, z1],
        [-hx,  hy, z1]])

    faces = np.array([[
        0,3,1],
        [1,3,2],
        [0,4,7],
        [0,7,3],
        [4,5,6],
        [4,6,7],
        [5,1,2],
        [5,2,6],
        [2,3,6],
        [3,7,6],
        [0,1,5],
        [0,5,4]])

    try:
        cube = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j],:]

        import tempfile
        temp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(temp_dir, name)
        blob_path = f"meshes/{name}"
        cube.save(tmp_path)

        if MESH_STORAGE == 'gcs':
            upload_to_gcs(bucket_name, tmp_path, blob_path)
            # Remove local temp file after upload
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            # Keep local file for direct download via /download-stl
            print(f"[INFO] Stored STL locally at {tmp_path}")
            # Record manifest to allow on-demand regeneration
            manifest = _load_manifest()
            manifest[name] = { 'amount': float(weight), 'density': float(density) }
            _save_manifest(manifest)
    except Exception as e:
        print(f"[WARN] STL generation/upload failed for {name}: {e}")

    return x, y, z

def recommend(gender, age, height, weight, carbohydrate, protein, fat, activity, diet, preference):
    rmr = calculate_rmr(weight, height, age, gender)
    calories = calculate_daily_calories(rmr, activity)
    diet_scale = [(0.50 / 4.1, 0.20 / 4.1, 0.30 / 8.8), # balanced
              (0.60 / 4.1, 0.20 / 4.1, 0.20 / 8.8), # low fat
              (0.20 / 4.1, 0.30 / 4.1, 0.50 / 8.8), # low carbs,
              (0.28 / 4.1, 0.39 / 4.1, 0.33 / 8.8)] # high protein

    carbohydrate_intake, protein_intake, fat_intake = (calories * i for i in diet_scale[diet])

    carbohydrate_needed = carbohydrate_intake - carbohydrate
    protein_needed = protein_intake - protein
    fat_needed = fat_intake - fat

    # Each row is [carbohydrates, proteins, fats]
    W_per_hundred = np.array([
        [17, 1.56, 0.05],  # PSP
        [11.2, 6.6, 0.61],   # Red Lentils
        [1.4, 1.38, 12.1],  # Avocado
        [0.06, 19.8, 1.15]   # Chicken Breast
    ])

    W = W_per_hundred * 0.01

    name = ['Purple Sweet Potato', 'Red Lentils', 'Avocado', 'Chicken Breast']
    density = [0.81, 1.182, 0.63, 0.82]
    
    if preference: blocked = 3
    else: blocked = 1

    y = np.array([carbohydrate_needed, protein_needed, fat_needed]) # [carbohydrates, proteins, fats]

    # positive_indices = np.where(y > 0)[0]
    # positive_y = y[positive_indices]
    # positive_y = y
    # positive_y[positive_y <= 0] = 0
    
    nonlinear_constraint = NonlinearConstraint(constraint_func, 0.01, np.inf)

    solutions = []
    best_candidate = None  # fallback if nothing meets tolerance

    # def fun(amounts, nutritional_matrix, target):
    #     amounts = amounts.reshape(-1, 1)
    #     total_nutrition = np.dot(nutritional_matrix.T, amounts)
    #     return np.linalg.norm(total_nutrition - target)

    upper_bound = 10           # set None for “no limit”
    tolerance   = 1e-6

    def solve_pair(selected_nutrition, positive_y):
        # A = np.array([[nutrients[name1][c], nutrients[name2][c]] for c in cols], dtype=float) #selected_nutrition
        # Least-squares solution (satisfies A @ x ≈ b in L2 sense)
        # x, residuals, _, _ = np.linalg.lstsq(selected_nutrition.T, positive_y, rcond=None)
        x, res_norm = nnls(selected_nutrition.T, positive_y)
        residual = np.linalg.norm(selected_nutrition.T @ x - positive_y, ord=1)     # total absolute error
        return x, residual

    if np.any(y > 0):
        mask = y > 0  
        for indices in combinations(range(4), 2):
            if (blocked in indices): continue
            selected_nutrition = W[list(indices)]

            fun = lambda x: np.linalg.norm(selected_nutrition.T[mask, :] @ x - y[mask])
            res = minimize(fun, np.zeros(len(indices)), method='L-BFGS-B', bounds=[(0., MAX_VOLUME / density[indices[x]]) for x in range(len(indices))])

            print(f"Testing combination {indices}: amounts={res.x}, error={res.fun}")
            # Track best candidate even if above tolerance
            if res.x[0] > 0 and res.x[1] > 0:
                if best_candidate is None or res.fun < best_candidate[2]:
                    best_candidate = (indices, res.x, res.fun)

            # Accept solution if both amounts are positive and error is reasonable
            if res.x[0] > 0 and res.x[1] > 0 and res.fun < TOLERANCE:
                solutions.append((indices, res.x, res.fun))
                print(f"  -> ACCEPTED")
            else:
                print(f"  -> REJECTED (tolerance={TOLERANCE})")

        # If none accepted, use best candidate so we always produce meshes
        if not solutions and best_candidate:
            solutions.append(best_candidate)
            print(f"\nNo solutions under tolerance. Using best available combination with error={best_candidate[2]:.2f}")

        solutions.sort(key=lambda x: x[2])
        print(f"\n=== Found {len(solutions)} valid solutions ===")
    
    # Limit number of solutions to avoid long runtimes / memory use
    solutions = solutions[:MAX_SOLUTIONS]
    results = []

    for index in range(len(solutions)):
        indices, amounts, norm = solutions[index]
        material_mesh_list = []
        carbohydrate_supplement = protein_supplement = fat_supplement = 0
        cumulative_z = 0.0  # running z offset so each food item stacks on top of the previous
        # print(amounts)
        for i in range(len(amounts)):             
            amounts[i] = round(amounts[i], 2)
            if amounts[i] == 0: continue
            mesh_name = str(index) + "_" + name[indices[i]] + ".stl"
            carbohydrate_supplement += amounts[i] * W[indices[i]][0]
            protein_supplement += amounts[i] * W[indices[i]][1]
            fat_supplement += amounts[i] * W[indices[i]][2]
            z_off = cumulative_z
            # Record manifest for on-demand regeneration, regardless of generation mode
            try:
                manifest = _load_manifest()
                manifest[mesh_name] = { 'amount': float(amounts[i]), 'density': float(density[indices[i]]), 'z_offset': float(z_off) }
                _save_manifest(manifest)
            except Exception as mf_err:
                print(f"[WARN] Failed to update manifest for {mesh_name}: {mf_err}")

            # Decide whether to generate STL based on MESH_MODE
            generate_mesh = (MESH_MODE == 'all') or (MESH_MODE == 'first' and index == 0)
            x, y, z = mesh_generation(mesh_name, amounts[i], density[indices[i]], z_offset=z_off) if generate_mesh else calculate_cube_dimension(amounts[i] / density[indices[i]])
            # Show download links when meshes are allowed; on-demand regen will be used if file is missing
            mesh_field = mesh_name if MESH_MODE != 'none' and x and y and z else ''
            if x and y and z:
                cumulative_z += z  # advance the stack by this item's thickness
                material_mesh_list.append({'name': name[indices[i]], 'mesh': mesh_field, 'gram': amounts[i], 'x': round(x, 2), 'y': round(y, 2), 'z': round(z, 2)})
        results.append((material_mesh_list, round(carbohydrate_supplement, 2), round(protein_supplement, 2), round(fat_supplement, 2)))            

    # print(results)

    recommend_dict = {'calories': round(calories, 2), 
                      'carbohydrate_intake': round(carbohydrate_intake, 2),
                      'protein_intake': round(protein_intake, 2),
                      'fat_intake': round(fat_intake, 2),
                      'carbohydrate_needed': round(carbohydrate_needed, 2),
                      'protein_needed': round(protein_needed, 2),
                      'fat_needed': round(fat_needed, 2),
                    #   'carbohydrate_supplement': round(carbohydrate_needed, 2),
                    #   'protein_supplement': round(protein_needed, 2),
                    #   'fat_supplement': round(fat_needed, 2),
                      'results': results
                    }
    
    return recommend_dict
 
@app.route('/data_collection', methods=["GET", "POST"])
def data_collection():
    # carbs = float(request.args.get('carbs'))
    # protein = float(request.args.get('protein'))
    # fat = float(request.args.get('fat'))
    if request.method == "POST":
        # print(request.form)
        submit = request.form["submit"]
        info_dict = {
            'gender': int(request.form["gender"]),
            'age': int(request.form["age"]),
            'height': float(request.form["height"]), 
            'weight': float(request.form["weight"]), 
            'carbs': float(request.form["carbohydrate"]),
            'protein': float(request.form["protein"]),
            'fat': float(request.form["fat"]),
            'activity': int(request.form["activity"]),
            'diet': int(request.form["diet"]),
            'preference': int(request.form["preference"]),
        }
        
        if submit: return redirect(url_for("nutrition_recommendation_display", info_dict=info_dict))
    # return render_template("data-collection.html", carbs=carbs, protein=protein, fat=fat)
    return render_template("data-collection.html")

def list2dict(info_dict):
    new_dict = {}
    for d in info_dict:
        index, value = d.split(':')

        if '{' in index: index = index[2:-1]
        else: index = index[2:-1]

        if '}' in value: value = value[:-1]
        else: value = value

        new_dict[index] = float(value)
    
    return new_dict

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob_path = os.path.join("/meshes", filename)
    blob = bucket.blob(blob_path)

    file_stream = io.BytesIO()
    blob.download_to_file(file_stream)
    file_stream.seek(0)

    return send_file(file_stream, as_attachment=True, download_name=filename)

@app.route('/nutrition_recommendation_display', methods=["GET", "POST"])
def nutrition_recommendation_display():
    info_dict = request.args.get('info_dict')
    
    # If no info_dict provided, use default values for demo
    if info_dict is None:
        # Use default values as a dict directly
        info_dict = {
            'gender': 0,
            'age': 25,
            'height': 170.0,
            'weight': 70.0,
            'carbs': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'activity': 2,
            'diet': 0,
            'preference': 0
        }
    else:
        info_dict = info_dict.split(",")
        info_dict = list2dict(info_dict)
    
    # print(info_dict)
    recommend_dict = recommend(int(info_dict['gender']), int(info_dict['age']), info_dict['height'], info_dict['weight'], info_dict['carbs'], \
                               info_dict['protein'], info_dict['fat'], int(info_dict['activity']), int(info_dict['diet']), int(info_dict['preference']))

    # print(recommend_dict)
    
    if request.method == "POST":
        # print(request.form)
        refresh = request.form["refresh"]
        # if refresh: return redirect("/upload_image")
        if refresh: return redirect("/data_collection")
    return render_template("nutrition-recommendation.html", recommend_dict=recommend_dict)

# Chatbot routes
@app.route('/chatbot', methods=["GET", "POST"])
def chatbot():
    """Serve the chatbot interface"""
    return render_template("chatbot.html")

@app.route('/api/search-food', methods=['POST'])
def api_search_food():
    """Search for food in CSV data and return parsed nutrition. Supports multiple foods separated by commas."""
    try:
        data = request.json
        food_input = data.get('food_input', '').strip()
        
        if not food_input:
            return jsonify({'error': 'No food input provided'}), 400
        
        # Split by comma variants and semicolons to handle multiple foods
        food_items = [item.strip() for item in re.split(r'[,，;]+', food_input)]
        
        # Store all nutrition data and individual results
        total_nutrition = {
            'carbs': 0,
            'protein': 0,
            'fat': 0,
            'calories': 0
        }
        individual_foods = []
        errors = []
        
        # Process each food item
        for food_item in food_items:
            if not food_item:
                continue

            # Handle direct macro entries in mixed input (e.g., "100g carb")
            direct_macro = parse_direct_macro_input(food_item)
            if direct_macro is not None:
                total_nutrition['carbs'] += direct_macro.get('carbs', 0)
                total_nutrition['protein'] += direct_macro.get('protein', 0)
                total_nutrition['fat'] += direct_macro.get('fat', 0)
                total_nutrition['calories'] += direct_macro.get('calories', 0)
                individual_foods.append(direct_macro)
                continue
                
            # Parse user input
            food_name, quantity, unit = parse_food_input(food_item)
            
            # Resolve nutrition with CSV-first + USDA fallback
            nutrition, source = get_food_nutrition_with_fallback(food_name, quantity, unit)
            if not nutrition:
                # Both CSV and USDA failed - ask user to search manually
                errors.append({
                    'food': food_item,
                    'message': f"'{food_item}' not found in local database or USDA API. Please search manually and enter nutrition values.",
                    'manual_input': True
                })
                continue
            
            if not nutrition:
                errors.append({
                    'food': food_item,
                    'message': f"Could not retrieve nutrition for '{food_item}'",
                    'manual_input': False
                })
                continue
            
            # Add source to nutrition data
            nutrition['source'] = source
            
            # Add to total nutrition
            total_nutrition['carbs'] += nutrition.get('carbs', 0)
            total_nutrition['protein'] += nutrition.get('protein', 0)
            total_nutrition['fat'] += nutrition.get('fat', 0)
            total_nutrition['calories'] += nutrition.get('calories', 0)
            
            # Store individual food info
            individual_foods.append({
                'food_name': nutrition.get('food_name', ''),
                'quantity': nutrition.get('quantity', 0),
                'unit': nutrition.get('unit', ''),
                'carbs': nutrition.get('carbs', 0),
                'protein': nutrition.get('protein', 0),
                'fat': nutrition.get('fat', 0),
                'calories': nutrition.get('calories', 0),
                'source': nutrition.get('source', 'Unknown')
            })
        
        # Check if we successfully processed at least one food
        if not individual_foods:
            if errors:
                # Separate errors into manual input needed vs other errors
                manual_input_foods = [e for e in errors if isinstance(e, dict) and e.get('manual_input')]
                other_errors = [e for e in errors if isinstance(e, dict) and not e.get('manual_input')]
                
                error_message = 'Could not find foods in any database'
                if manual_input_foods:
                    error_message += '. Please search the internet and manually enter nutrition values for: ' + ', '.join([e['food'] for e in manual_input_foods])
                
                return jsonify({
                    'error': error_message,
                    'all_errors': errors,
                    'manual_input_required': len(manual_input_foods) > 0,
                    'manual_input_foods': manual_input_foods
                }), 404
            return jsonify({
                'error': 'No valid food items provided',
                'suggestion': 'Please provide at least one food item'
            }), 400
        
        # Build response
        response = {
            'success': True,
            'nutrition': {
                'carbs': round(total_nutrition['carbs'], 2),
                'protein': round(total_nutrition['protein'], 2),
                'fat': round(total_nutrition['fat'], 2),
                'calories': round(total_nutrition['calories'], 2)
            },
            'individual_foods': individual_foods,
            'original_input': food_input,
            'foods_processed': len(individual_foods)
        }
        
        # Include warnings if some items failed
        if errors:
            manual_input_foods = [e for e in errors if isinstance(e, dict) and e.get('manual_input')]
            response['warnings'] = errors
            if manual_input_foods:
                response['manual_input_required'] = True
                response['manual_input_foods'] = manual_input_foods
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in api_search_food: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate-recommendation', methods=['POST'])
def api_calculate_recommendation():
    """Calculate nutrition recommendation based on user info and daily intake"""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        user_info = data.get('user_info', {})
        daily_nutrition = data.get('daily_nutrition', {})

        if not data:
            return jsonify({'error': 'Request body missing. Send JSON with user_info and daily_nutrition.'}), 400

        def to_int(val, default=0):
            try:
                # Treat "" or None as missing -> default
                if val is None or val == '':
                    return default
                return int(val)
            except Exception:
                return default

        def to_float(val, default=0.0):
            try:
                if val is None or val == '':
                    return default
                return float(val)
            except Exception:
                return default

        # Extract values with safe coercion
        gender = to_int(user_info.get('gender'), 0)
        age = to_int(user_info.get('age'), 0)
        height = to_float(user_info.get('height'), 0)
        weight = to_float(user_info.get('weight'), 0)
        carbs = to_float(daily_nutrition.get('carbs'), 0)
        protein = to_float(daily_nutrition.get('protein'), 0)
        fat = to_float(daily_nutrition.get('fat'), 0)
        activity = to_int(user_info.get('activity'), 0)
        diet = to_int(user_info.get('diet'), 0)
        preference = to_int(user_info.get('preference'), 0)

        # Minimal validation with soft defaults
        missing = []
        if gender not in [0, 1]: missing.append('gender')
        if age <= 0: missing.append('age')
        if height <= 0: missing.append('height')
        if weight <= 0: missing.append('weight')
        if activity not in [0, 1, 2, 3]: missing.append('activity')
        if diet not in [0, 1, 2, 3]: missing.append('diet')
        if preference not in [0, 1]: missing.append('preference')

        # If missing, apply sensible defaults to keep API responsive
        if missing:
            defaults = {
                'gender': 0,
                'age': 25,
                'height': 170.0,
                'weight': 70.0,
                'activity': 2,
                'diet': 0,
                'preference': 0,
            }
            gender = gender if gender in [0,1] else defaults['gender']
            age = age if age > 0 else defaults['age']
            height = height if height > 0 else defaults['height']
            weight = weight if weight > 0 else defaults['weight']
            activity = activity if activity in [0,1,2,3] else defaults['activity']
            diet = diet if diet in [0,1,2,3] else defaults['diet']
            preference = preference if preference in [0,1] else defaults['preference']
            note = f"Applied defaults for: {', '.join(missing)}"
        else:
            note = None
        
        # Call existing recommendation function with a robust fallback
        try:
            recommend_dict = recommend(gender, age, height, weight, carbs, protein, fat, activity, diet, preference)
        except Exception as rec_err:
            # Fallback: compute targets and needs without optimization/meshes
            try:
                rmr = calculate_rmr(weight, height, age, gender)
                calories = calculate_daily_calories(rmr, activity)
                diet_scale = [
                    (0.50 / 4.1, 0.20 / 4.1, 0.30 / 8.8),  # balanced
                    (0.60 / 4.1, 0.20 / 4.1, 0.20 / 8.8),  # low fat
                    (0.20 / 4.1, 0.30 / 4.1, 0.50 / 8.8),  # low carbs
                    (0.28 / 4.1, 0.39 / 4.1, 0.33 / 8.8),  # high protein
                ]
                carbohydrate_intake, protein_intake, fat_intake = (calories * i for i in diet_scale[diet])
                recommend_dict = {
                    'calories': round(calories, 2),
                    'carbohydrate_intake': round(carbohydrate_intake, 2),
                    'protein_intake': round(protein_intake, 2),
                    'fat_intake': round(fat_intake, 2),
                    'carbohydrate_needed': round(carbohydrate_intake - carbs, 2),
                    'protein_needed': round(protein_intake - protein, 2),
                    'fat_needed': round(fat_intake - fat, 2),
                    'results': []
                }
                note = 'Generated minimal recommendation (optimization failed)'
                if DIAG_MODE:
                    recommend_dict.update({'error': str(rec_err)})
            except Exception as fb_err:
                print(f"Fallback generation failed: {fb_err}")
                if DIAG_MODE:
                    return jsonify({'error': f'Fallback failed: {fb_err}'}), 500
                raise rec_err

        if note:
            recommend_dict['note'] = note
        
        saved_record = save_user_record(
            user_id=user_id,
            user_info={
                'gender': gender,
                'age': age,
                'height': height,
                'weight': weight,
                'activity': activity,
                'diet': diet,
                'preference': preference,
            },
            daily_nutrition={
                'carbs': carbs,
                'protein': protein,
                'fat': fat,
            },
            recommendation=recommend_dict
        )

        return jsonify({
            'success': True,
            'user_id': saved_record.get('user_id'),
            'recommendation': recommend_dict
        }), 200
    
    except Exception as e:
        print(f"Error in api_calculate_recommendation: {e}")
        if DIAG_MODE:
            return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Recommendation failed. Please try again later.'}), 500

@app.route('/api/user-records', methods=['GET', 'POST'])
def api_user_records():
    """Create/list backend user records for multi-user support."""
    try:
        if request.method == 'GET':
            records = _load_user_records()
            # Return full records for UI usage (including user_info for display)
            summaries = []
            for _, record in records.items():
                summaries.append({
                    'id': record.get('user_id'),
                    'user_id': record.get('user_id'),
                    'created_at': record.get('created_at'),
                    'updated_at': record.get('updated_at'),
                    'user_info': record.get('user_info', {}),
                    'history_count': len(record.get('history', []))
                })
            summaries.sort(key=lambda r: r.get('updated_at', ''), reverse=True)
            return jsonify({'success': True, 'records': summaries}), 200

        data = request.json or {}
        
        # Handle creation with just a name (from multi-user UI)
        if 'name' in data and 'user_id' not in data:
            user_id = str(uuid.uuid4())
            user_info = {'name': data.get('name')}
            saved = save_user_record(user_id=user_id, user_info=user_info)
            return jsonify({
                'success': True, 
                'user': {
                    'id': user_id,
                    'user_id': user_id,
                    'created_at': saved.get('created_at'),
                    'updated_at': saved.get('updated_at'),
                    'user_info': user_info
                }
            }), 200
        
        # Handle normal case (full user record update)
        user_id = data.get('user_id')
        user_info = data.get('user_info', {})
        saved = save_user_record(user_id=user_id, user_info=user_info)
        return jsonify({'success': True, 'record': saved}), 200
    except Exception as e:
        print(f"Error in api_user_records: {e}")
        if DIAG_MODE:
            return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'User record operation failed.'}), 500

@app.route('/api/user-records/<user_id>', methods=['GET', 'DELETE'])
def api_user_record_detail(user_id):
    """Fetch one full user record including history, or delete a user."""
    try:
        if request.method == 'DELETE':
            # Delete user record
            records = _load_user_records()
            if user_id not in records:
                return jsonify({'error': 'User record not found'}), 404
            
            del records[user_id]
            _save_user_records(records)
            return jsonify({'success': True, 'message': 'User deleted'}), 200
        
        # GET request
        record = get_user_record(user_id)
        if not record:
            return jsonify({'error': 'User record not found'}), 404
        return jsonify({'success': True, 'record': record}), 200
    except Exception as e:
        print(f"Error in api_user_record_detail: {e}")
        if DIAG_MODE:
            return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Unable to fetch user record.'}), 500

@app.route('/download-stl/<path:filename>', methods=['GET'])
def download_stl(filename):
    """Download STL file from Google Cloud Storage"""
    try:
        print(f"[DEBUG] Attempting to download STL file: {filename}")
        if MESH_STORAGE == 'local':
            import tempfile
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, filename)
            if not os.path.exists(local_path):
                print(f"[DEBUG] Local STL not found at {local_path}, attempting regeneration from manifest")
                # Try on-demand regeneration if manifest has info
                manifest = _load_manifest()
                meta = manifest.get(filename)
                if meta and 'amount' in meta and 'density' in meta:
                    try:
                        # Regenerate STL file (restore z_offset so stacking is preserved)
                        mesh_generation(filename, float(meta['amount']), float(meta['density']), z_offset=float(meta.get('z_offset', 0.0)))
                    except Exception as regen_err:
                        print(f"[WARN] Regeneration failed: {regen_err}")
                else:
                    print("[DEBUG] No manifest entry for this file; cannot regenerate")
                # Re-check existence after regeneration attempt
                if not os.path.exists(local_path):
                    return jsonify({'error': f'File not found (local): {filename}'}), 404
            with open(local_path, 'rb') as f:
                file_data = io.BytesIO(f.read())
            file_data.seek(0)
            print(f"[DEBUG] Serving local STL {filename}, size: {file_data.getbuffer().nbytes} bytes")
            return send_file(
                file_data,
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=filename
            )
        else:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"meshes/{filename}"
            print(f"[DEBUG] GCS blob path: {blob_path}")
            blob = bucket.blob(blob_path)
            
            # Check if blob exists
            if not blob.exists():
                print(f"[DEBUG] Blob does not exist at path: {blob_path}")
                return jsonify({'error': f'File not found in storage: {filename}'}), 404
            
            # Download to memory
            file_data = io.BytesIO()
            blob.download_to_file(file_data)
            file_data.seek(0)
            print(f"[DEBUG] Successfully downloaded {filename}, size: {file_data.getbuffer().nbytes} bytes")
            
            return send_file(
                file_data,
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=filename
            )
    except Exception as e:
        print(f"[ERROR] Error downloading STL: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'File not found or download failed: {str(e)}'}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/daily-intake/<user_id>', methods=['GET', 'POST'])
def save_daily_intake(user_id):
    """Save or retrieve daily nutrition history for a user."""
    try:
        records = _load_user_records()
        if user_id not in records:
            return jsonify({'error': 'User not found'}), 404

        if request.method == 'GET':
            history = records[user_id].get('daily_history', [])
            return jsonify({'success': True, 'history': history}), 200

        data = request.get_json() or {}
        daily_nutrition = data.get('daily_nutrition', {})
        recommended = data.get('recommended', {})
        today = datetime.now().strftime('%Y-%m-%d')

        user = records[user_id]
        daily_history = user.setdefault('daily_history', [])

        # Update existing entry for today or append a new one
        today_entry = next((e for e in daily_history if e.get('date') == today), None)
        if today_entry:
            today_entry['nutrition'] = daily_nutrition
            if recommended:
                today_entry['recommended'] = recommended
            today_entry['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        else:
            entry = {
                'date': today,
                'nutrition': daily_nutrition,
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }
            if recommended:
                entry['recommended'] = recommended
            daily_history.append(entry)

        _save_user_records(records)
        return jsonify({'success': True, 'user_id': user_id}), 200
    except Exception as e:
        print(f"[ERROR] Failed to save/get daily intake: {e}")
        return jsonify({'error': str(e)}), 500
 
# main driver function
if __name__ == '__main__':
    # Run on 0.0.0.0 to allow external access (Cloudflare tunnel, network access, etc.)
    app.run(host="0.0.0.0", port=5000, debug=True)