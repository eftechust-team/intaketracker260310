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
import struct
import zipfile
import html as _html
import xml.etree.ElementTree as ET
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
CSV_NUTRITION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "FoodSAM", "food_full_data_revised.csv"))
csv_data = []
csv_loaded = False
csv_mtime = None

# Simple cache to reduce repeated lookups
_search_cache = {}
_nutrition_cache = {}

def load_nutrition_csv():
    """Load the nutrition data from CSV file into memory"""
    global csv_data, csv_loaded, csv_mtime, _search_cache, _nutrition_cache
    try:
        csv_path = CSV_NUTRITION_PATH
        if os.path.exists(csv_path):
            csv_data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data.append(row)
            csv_loaded = True
            csv_mtime = os.path.getmtime(csv_path)
            # Invalidate caches so lookups use refreshed CSV content.
            _search_cache.clear()
            _nutrition_cache.clear()
            print(f"Loaded nutrition data from CSV: {len(csv_data)} food items")
            return True
        else:
            print(f"CSV file not found at {csv_path}")
            return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False


def ensure_nutrition_csv_fresh():
    """Reload CSV if the source file changed on disk."""
    global csv_mtime
    try:
        if not os.path.exists(CSV_NUTRITION_PATH):
            return
        current_mtime = os.path.getmtime(CSV_NUTRITION_PATH)
        if (not csv_loaded) or (csv_mtime is None) or (current_mtime > csv_mtime):
            load_nutrition_csv()
    except Exception as e:
        print(f"[WARN] Could not refresh CSV data: {e}")

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


def _singularize(word):
    """Convert a single English word to its approximate singular base form."""
    if len(word) <= 2:
        return word
    # ies → y  (berries→berry, fries→fry, cranberries→cranberry)
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    # Explicit sibilant-ending plurals: strip 'es'
    # sses→ss, shes→sh, ches→ch, xes→x, zes→z  (e.g., peaches→peach, boxes→box)
    if (word.endswith('sses') or word.endswith('shes') or
            word.endswith('ches') or word.endswith('xes') or word.endswith('zes')):
        return word[:-2]
    # oes → o  (tomatoes→tomato, potatoes→potato)
    if word.endswith('oes') and len(word) > 4:
        return word[:-2]
    # For other 'es' endings: if stripping just 's' leaves a word ending in 'e',
    # that 'e' was part of the base (noodles→noodle, olives→olive, grapes→grape)
    if word.endswith('es') and len(word) > 3:
        stem_s = word[:-1]   # strip just 's' → keeps trailing 'e'
        if stem_s.endswith('e'):
            return stem_s    # noodles→noodle ✓
        return word[:-2]     # fallback: strip 'es'
    # Plain 's' plural (beans→bean, shoots→shoot, dumplings→dumpling, peas→pea)
    if word.endswith('s') and len(word) > 2:
        return word[:-1]
    return word


def normalize_food_name(name):
    """Normalize food names to their singular base form for better CSV matching.

    Applies per-word singularization so that multi-word names work correctly,
    e.g. 'green beans' → 'green bean', 'wonton dumplings' → 'wonton dumpling',
    'noodles' → 'noodle', 'dried cranberries' → 'dried cranberry'.
    """
    cleaned = (name or '').strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return ' '.join(_singularize(w) for w in cleaned.split())


def parse_direct_macro_input(text):
    """Parse direct nutrition input like '100g carb', '-20 fat', '1000kcal', or '-100 kcal'."""
    cleaned = (text or '').strip()

    macro_match = re.match(
        r'^([+-]?\d+(?:\.\d+)?)\s*g?\s*(carb|carbon|carbohydrate|protein|fat)s?$',
        cleaned,
        re.IGNORECASE,
    )
    calorie_match = re.match(
        r'^([+-]?\d+(?:\.\d+)?)\s*(kcal|cal|calorie|calories)$',
        cleaned,
        re.IGNORECASE,
    )

    if not macro_match and not calorie_match:
        return None

    nutrition = {'carbs': 0.0, 'protein': 0.0, 'fat': 0.0, 'calories': 0.0}

    if macro_match:
        amount = float(macro_match.group(1))
        macro_type = macro_match.group(2).lower()
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
            'calories': 0.0,
        }

    amount = float(calorie_match.group(1))
    nutrition['calories'] = amount
    return {
        'food_name': 'direct calories',
        'quantity': amount,
        'unit': 'kcal',
        'carbs': 0.0,
        'protein': 0.0,
        'fat': 0.0,
        'calories': round(nutrition['calories'], 2),
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
    ensure_nutrition_csv_fresh()

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
    
    # Extract nutrition values from CSV (values are per gram).
    # The CSV uses short lowercase headers: calories, protein, carbohydrates, fat, density.
    def _csv_float(row, *keys, default=0.0):
        """Try each key in order and return the first non-empty float value found."""
        for key in keys:
            raw = str(row.get(key, '') or '').strip()
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    continue
        return default

    try:
        calories_per_gram = _csv_float(
            csv_food_row,
            'calories', 'Calories', 'Calories (kcal/g)', 'calories (kcal/g)',
        )
        protein_per_gram = _csv_float(
            csv_food_row,
            'protein', 'Protein', 'Protein (g/g)', 'protein (g/g)',
        )
        carbs_per_gram = _csv_float(
            csv_food_row,
            'carbohydrates', 'Carbohydrates', 'carbs', 'Carbs',
            'Carbohydrates (g/g)', 'carbohydrates (g/g)',
        )
        fat_per_gram = _csv_float(
            csv_food_row,
            'fat', 'Fat', 'Fat (g/g)', 'fat (g/g)',
        )
        density = _csv_float(
            csv_food_row,
            'density', 'Density', 'Density (g/ml)', 'density (g/ml)',
            default=1.0,
        )
        if density == 0.0:
            density = 1.0
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
        prompt = (
            f"Estimate the nutrition for {quantity}{unit} {food_name}. "
            "Return ONLY valid JSON with numeric values using this exact schema: "
            '{"calories": 0, "carbs": 0, "protein": 0, "fat": 0}. '
            "Calories must be in kcal. Carbs, protein, and fat must be in grams. "
            "Do not include explanations, markdown, code fences, or any extra text."
        )
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOUBAO_API_KEY}"
        }
        
        payload = {
            "model": DOUBAO_MODEL,
            "messages": [
                {"role": "system", "content": "你是一个营养学家助手，专门提供食物的营养信息。你必须只返回JSON，字段必须包含calories、carbs、protein、fat，全部为数字。"},
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
            
            nutrition = {
                'carbs': 0.0,
                'protein': 0.0,
                'fat': 0.0,
                'calories': 0.0
            }

            def _coerce_number(value):
                if value is None:
                    return 0.0
                if isinstance(value, (int, float)):
                    return float(value)
                text = str(value).strip()
                match = re.search(r'([\d.]+)', text)
                if match:
                    try:
                        return float(match.group(1))
                    except Exception:
                        return 0.0
                return 0.0

            def _extract_number(patterns, text):
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            return float(match.group(1))
                        except Exception:
                            continue
                return 0.0

            # Prefer strict JSON if Doubao follows the instruction.
            cleaned = content.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    nutrition['calories'] = _coerce_number(
                        parsed.get('calories', parsed.get('calorie', parsed.get('energy', parsed.get('kcal', parsed.get('热量', parsed.get('能量', 0))))))
                    )
                    nutrition['carbs'] = _coerce_number(
                        parsed.get('carbs', parsed.get('carbohydrates', parsed.get('carbohydrate', parsed.get('碳水', parsed.get('碳水化合物', 0)))))
                    )
                    nutrition['protein'] = _coerce_number(
                        parsed.get('protein', parsed.get('proteins', parsed.get('蛋白质', 0)))
                    )
                    nutrition['fat'] = _coerce_number(
                        parsed.get('fat', parsed.get('fats', parsed.get('脂肪', 0)))
                    )
                except Exception:
                    pass

            # Fallback to flexible text parsing for English/Chinese/energy wording.
            if nutrition['calories'] <= 0:
                nutrition['calories'] = _extract_number([
                    r'calories?[^\d]{0,12}([\d.]+)',
                    r'energy[^\d]{0,12}([\d.]+)',
                    r'(?:热量|能量)[^\d]{0,12}([\d.]+)',
                    r'([\d.]+)\s*kcal',
                ], cleaned)
            if nutrition['carbs'] <= 0:
                nutrition['carbs'] = _extract_number([
                    r'carbs?[^\d]{0,12}([\d.]+)',
                    r'carbohydrates?[^\d]{0,12}([\d.]+)',
                    r'(?:碳水|碳水化合物)[^\d]{0,12}([\d.]+)',
                ], cleaned)
            if nutrition['protein'] <= 0:
                nutrition['protein'] = _extract_number([
                    r'protein[^\d]{0,12}([\d.]+)',
                    r'蛋白质[^\d]{0,12}([\d.]+)',
                ], cleaned)
            if nutrition['fat'] <= 0:
                nutrition['fat'] = _extract_number([
                    r'fat[^\d]{0,12}([\d.]+)',
                    r'脂肪[^\d]{0,12}([\d.]+)',
                ], cleaned)

            if nutrition['calories'] <= 0 and _is_nutrition_meaningful(nutrition):
                nutrition['calories'] = nutrition['carbs'] * 4 + nutrition['protein'] * 4 + nutrition['fat'] * 9
            
            if _is_nutrition_meaningful({'carbs': nutrition['carbs'], 'protein': nutrition['protein'], 'fat': nutrition['fat']}):
                result = {
                    'food_name': food_name,
                    'carbs': round(nutrition['carbs'], 2),
                    'protein': round(nutrition['protein'], 2),
                    'fat': round(nutrition['fat'], 2),
                    'calories': round(nutrition['calories'], 2),
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

    # USDA nutrientNumber reference in FoodData Central:
    # 208=Energy (kcal), 203=Protein, 205=Carbohydrate, 204=Total lipid (fat)
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

        nutrient_id = str(
            nutrient_obj.get('id')
            or nutrient.get('nutrientId')
            or ''
        ).strip()

        unit_name = str(
            nutrient_obj.get('unitName')
            or nutrient.get('unitName')
            or ''
        ).strip().lower()

        amount = nutrient.get('amount')
        if amount is None:
            amount = nutrient.get('value', 0)

        try:
            amount = float(amount or 0)
        except Exception:
            amount = 0.0

        if nutrient_number == '208' or nutrient_id == '1008' or ('energy' in nutrient_name and unit_name == 'kcal'):
            nutrition_per_100g['calories'] = amount
        elif nutrient_number == '203' or nutrient_id == '1003' or 'protein' in nutrient_name:
            nutrition_per_100g['protein'] = amount
        elif nutrient_number == '205' or nutrient_id == '1005' or ('carbohydrate' in nutrient_name and 'fiber' not in nutrient_name):
            nutrition_per_100g['carbs'] = amount
        elif nutrient_number == '204' or nutrient_id == '1004' or ('fat' in nutrient_name and ('total' in nutrient_name or 'lipid' in nutrient_name)):
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

MISSING_FOODS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missing_foods.txt")

def _log_missing_food(food_name, quantity, unit, source, nutrition):
    """Append a food not found in CSV to missing_foods.txt."""
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        carbs = round(float(nutrition.get('carbs', 0)), 2) if nutrition else 'N/A'
        protein = round(float(nutrition.get('protein', 0)), 2) if nutrition else 'N/A'
        fat = round(float(nutrition.get('fat', 0)), 2) if nutrition else 'N/A'
        calories = round(float(nutrition.get('calories', 0)), 2) if nutrition else 'N/A'
        line = (
            f"[{timestamp}] food={food_name!r} | qty={quantity}{unit} | "
            f"source={source} | carbs={carbs}g protein={protein}g fat={fat}g calories={calories}kcal\n"
        )
        with open(MISSING_FOODS_LOG, 'a', encoding='utf-8') as f:
            f.write(line)
        print(f"[MissingFoodLog] Logged: {food_name!r} (source: {source})")
    except Exception as e:
        print(f"[MissingFoodLog] Failed to write log: {e}")


def get_food_nutrition_with_fallback(food_name, quantity, unit):
    """
    Resolve nutrition for a food item using a three-tier fallback:
      1. Local CSV database
      2. USDA FoodData Central API
      3. Doubao LLM
    Foods not found in CSV are recorded in missing_foods.txt.
    Returns tuple: (nutrition_dict_or_none, source_label)
    """
    # --- 1. CSV ---
    csv_results = search_csv_food(food_name)
    if csv_results and csv_results.get('foods'):
        for food_item in csv_results['foods']:
            csv_row = food_item.get('_csv_data')
            if not csv_row:
                continue
            fdc_id = food_item.get('fdcId', '')
            nutrition = get_food_nutrition_csv(fdc_id, csv_row, quantity, unit)
            if nutrition and _is_nutrition_meaningful(nutrition):
                print(f"[Fallback] Found in CSV for '{food_name}'")
                return nutrition, "CSV"

    # Not in CSV — will log regardless of where it's eventually resolved.

    # --- 2. USDA API ---
    usda_results = search_usda_food(food_name)
    if usda_results and usda_results.get('foods'):
        for food_item in usda_results['foods'][:5]:
            fdc_id = food_item.get('fdcId')
            if not fdc_id:
                continue
            nutrition = get_food_nutrition_usda(fdc_id, quantity, unit)
            if nutrition and _is_nutrition_meaningful(nutrition):
                print(f"[Fallback] Found via USDA API for '{food_name}'")
                _log_missing_food(food_name, quantity, unit, "USDA API", nutrition)
                return nutrition, "USDA API"

    # --- 3. Doubao LLM ---
    nutrition = query_doubao_nutrition(food_name, quantity, unit)
    if nutrition and _is_nutrition_meaningful(nutrition):
        print(f"[Fallback] Found via Doubao LLM for '{food_name}'")
        _log_missing_food(food_name, quantity, unit, "Doubao LLM", nutrition)
        return nutrition, "Doubao LLM"

    # Not found anywhere — still log it
    _log_missing_food(food_name, quantity, unit, "NOT FOUND", None)
    return None, ""


DIET_SCALE = [
    (0.50 / 4.1, 0.20 / 4.1, 0.30 / 8.8),
    (0.60 / 4.1, 0.20 / 4.1, 0.20 / 8.8),
    (0.20 / 4.1, 0.30 / 4.1, 0.50 / 8.8),
    (0.28 / 4.1, 0.39 / 4.1, 0.33 / 8.8),
]


def calculate_macro_targets(gender, age, height, weight, carbohydrate, protein, fat, activity, diet):
    rmr = calculate_rmr(weight, height, age, gender)
    calories = calculate_daily_calories(rmr, activity)
    carbohydrate_intake, protein_intake, fat_intake = (calories * i for i in DIET_SCALE[diet])
    carbohydrate_needed = carbohydrate_intake - carbohydrate
    protein_needed = protein_intake - protein
    fat_needed = fat_intake - fat
    return {
        'calories': round(calories, 2),
        'carbohydrate_intake': round(carbohydrate_intake, 2),
        'protein_intake': round(protein_intake, 2),
        'fat_intake': round(fat_intake, 2),
        'carbohydrate_needed': round(carbohydrate_needed, 2),
        'protein_needed': round(protein_needed, 2),
        'fat_needed': round(fat_needed, 2),
        'need_vector': np.array([carbohydrate_needed, protein_needed, fat_needed], dtype=float),
    }


def _looks_non_veg_name(food_name_text):
    n = (food_name_text or '').lower()
    tags = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'mutton', 'lamb', 'meat', 'tuna', 'salmon']
    return any(t in n for t in tags)


def _looks_snack_or_dessert_name(food_name_text):
    n = (food_name_text or '').lower()
    blocked = ['candy', 'chocolate', 'cake', 'cookie', 'soda', 'syrup', 'chips', 'popcorn', 'cracker', 'biscuit', 'snack']
    return any(t in n for t in blocked)


def resolve_recipe_food_candidates(food_names, preference):
    resolved = []
    unresolved = []
    seen = set()

    for raw_name in food_names:
        cleaned = (raw_name or '').strip()
        if not cleaned:
            continue
        key = normalize_food_name(cleaned)
        if key in seen:
            continue
        seen.add(key)

        nutrition, source = get_food_nutrition_with_fallback(cleaned, 100, 'g')
        if not nutrition or not _is_nutrition_meaningful(nutrition):
            unresolved.append(cleaned)
            continue

        resolved_name = nutrition.get('food_name') or cleaned
        if preference and _looks_non_veg_name(resolved_name):
            unresolved.append(cleaned)
            continue

        vec = np.array([
            float(nutrition.get('carbs', 0) or 0) / 100.0,
            float(nutrition.get('protein', 0) or 0) / 100.0,
            float(nutrition.get('fat', 0) or 0) / 100.0,
        ], dtype=float)
        if np.sum(vec) <= 1e-8:
            unresolved.append(cleaned)
            continue

        resolved.append({
            'name': resolved_name,
            'input_name': cleaned,
            'source': source,
            'vec': vec,
        })

    return resolved, unresolved


def suggest_foods_for_deficit(deficit_vec, excluded_names=None, preference=0, limit=6):
    ensure_nutrition_csv_fresh()

    excluded = {normalize_food_name(x) for x in (excluded_names or [])}
    suggestions = []

    def row_float(row, *keys, default=0.0):
        for key in keys:
            raw = str(row.get(key, '') or '').strip()
            if not raw:
                continue
            try:
                return float(raw)
            except Exception:
                continue
        return default

    for row in (csv_data or []):
        fname = (row.get('category_name') or '').strip()
        if not fname:
            continue
        norm = normalize_food_name(fname)
        if norm in excluded or norm == 'background':
            continue
        if preference and _looks_non_veg_name(fname):
            continue
        if _looks_snack_or_dessert_name(fname):
            continue

        vec = np.array([
            row_float(row, 'carbohydrates', 'Carbohydrates', 'carbs', 'Carbs', default=0.0),
            row_float(row, 'protein', 'Protein', default=0.0),
            row_float(row, 'fat', 'Fat', default=0.0),
        ], dtype=float)
        if np.sum(vec) <= 1e-8:
            continue

        score = float(np.dot(vec * 100.0, np.maximum(deficit_vec, 0.0)))
        if score > 0:
            suggestions.append((score, fname))

    suggestions.sort(key=lambda x: x[0], reverse=True)
    result = []
    seen = set()
    for _, fname in suggestions:
        norm = normalize_food_name(fname)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(fname)
        if len(result) >= limit:
            break
    return result


def build_recipe_option(title, candidates, target_need, use_all_requested=False):
    if not candidates:
        return None

    positive_target = np.maximum(target_need, 0.0)
    nutr = np.array([c['vec'] for c in candidates], dtype=float)
    count = len(candidates)
    min_grams = np.array([15.0 if use_all_requested else 0.0] * count, dtype=float)
    max_grams = np.array([350.0] * count, dtype=float)
    x0 = np.array([max(60.0, min_grams[i]) for i in range(count)], dtype=float)

    def objective(extra_amounts):
        grams = min_grams + np.maximum(extra_amounts, 0.0)
        supplied = np.dot(grams, nutr)
        under = np.maximum(positive_target - supplied, 0.0)
        over = np.maximum(supplied - positive_target, 0.0)
        return float(np.sum(under ** 2) + 4.0 * np.sum(over ** 2) + 0.0008 * np.sum(grams))

    try:
        res = minimize(
            objective,
            np.maximum(x0 - min_grams, 0.0),
            method='L-BFGS-B',
            bounds=[(0.0, max_grams[i] - min_grams[i]) for i in range(count)]
        )
        extra = res.x if res.success else np.maximum(x0 - min_grams, 0.0)
    except Exception:
        extra = np.maximum(x0 - min_grams, 0.0)

    grams = np.clip(min_grams + np.maximum(extra, 0.0), min_grams, max_grams)
    supplied = np.dot(grams, nutr)
    under = np.maximum(positive_target - supplied, 0.0)
    over = np.maximum(supplied - positive_target, 0.0)

    foods = []
    for idx, candidate in enumerate(candidates):
        gram = round(float(grams[idx]), 2)
        if gram <= 0:
            continue
        foods.append({
            'name': candidate['name'],
            'gram': gram,
            'source': candidate.get('source', ''),
        })

    if not foods:
        return None

    return {
        'title': title,
        'uses_all_requested': use_all_requested,
        'foods': foods,
        'supplied': {
            'carbs': round(float(supplied[0]), 2),
            'protein': round(float(supplied[1]), 2),
            'fat': round(float(supplied[2]), 2),
        },
        'shortfall_total': round(float(np.sum(under)), 2),
        'exceed_total': round(float(np.sum(over)), 2),
        'score': round(float(np.sum(under) + 2.5 * np.sum(over)), 3),
    }


def build_custom_recipe_recommendations(food_text, target_need, preference, limit=4):
    requested_foods = [item.strip() for item in re.split(r'[,，;]+', str(food_text or '')) if item.strip()]
    requested_foods = requested_foods[:6]
    resolved, unresolved = resolve_recipe_food_candidates(requested_foods, preference)

    recipes = []
    if resolved:
        recipe1 = build_recipe_option('Recipe 1', resolved, target_need, use_all_requested=True)
        if recipe1:
            recipes.append(recipe1)

        subset_candidates = []
        for subset_size in range(1, len(resolved)):
            for idx_tuple in combinations(range(len(resolved)), subset_size):
                chosen = [resolved[i] for i in idx_tuple]
                recipe = build_recipe_option('', chosen, target_need, use_all_requested=False)
                if recipe:
                    subset_candidates.append(recipe)

        subset_candidates.sort(key=lambda r: (r['score'], r['exceed_total'], r['shortfall_total']))
        seen = set()
        for recipe in subset_candidates:
            key = tuple(sorted(f['name'] for f in recipe['foods']))
            if key in seen:
                continue
            seen.add(key)
            recipe['title'] = f"Recipe {len(recipes) + 1}"
            recipes.append(recipe)
            if len(recipes) >= limit:
                break

    advice = {'message': '', 'suggested_foods': [], 'unresolved_foods': unresolved}
    if unresolved:
        advice['message'] = 'Some requested foods could not be resolved from CSV/USDA/Doubao.'

    if recipes:
        deficit = np.maximum(target_need - np.array([
            float(recipes[0]['supplied']['carbs']),
            float(recipes[0]['supplied']['protein']),
            float(recipes[0]['supplied']['fat']),
        ]), 0.0)
        if np.sum(deficit) > 12.0 or unresolved:
            advice['suggested_foods'] = suggest_foods_for_deficit(deficit, [f['name'] for f in resolved], preference, limit=6)
            if advice['suggested_foods'] and not advice['message']:
                advice['message'] = 'Your requested foods alone may not fully meet the supplementary nutrition.'
    elif requested_foods:
        advice['message'] = 'Could not build recipes from the requested foods.'
        advice['suggested_foods'] = suggest_foods_for_deficit(target_need, [], preference, limit=6)

    return {
        'requested_foods': requested_foods,
        'resolved_foods': [r['name'] for r in resolved],
        'unresolved_foods': unresolved,
        'recipes': recipes,
        'advice': advice,
    }

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


def _stl_to_triangles(stl_path):
    """Read a binary STL and return list of (v0,v1,v2) triangle tuples (mm)."""
    triangles = []
    try:
        with open(stl_path, 'rb') as f:
            f.read(80)  # header
            count = struct.unpack('<I', f.read(4))[0]
            for _ in range(count):
                f.read(12)  # normal
                v0 = struct.unpack('<fff', f.read(12))
                v1 = struct.unpack('<fff', f.read(12))
                v2 = struct.unpack('<fff', f.read(12))
                f.read(2)  # attr
                triangles.append((v0, v1, v2))
    except Exception as e:
        print(f'[WARN] _stl_to_triangles failed for {stl_path}: {e}')
    return triangles


def create_obj_bundle(stl_paths_and_names, output_obj_path):
    """
    Bundle multiple STL files into a single Wavefront OBJ file.
    stl_paths_and_names: list of (stl_file_path, object_name) tuples.
    Each STL becomes a named 'o' group; vertex indices are global and
    1-based as required by the OBJ spec.
    OBJ is plain text, requires no packaging, and is universally
    supported by all major slicers (PrusaSlicer, Cura, Bambu Studio,
    Blender, etc.).
    """
    lines = ['# ElevateFoods AI Nutrition Chatbot – multi-part food mesh\n']
    vertex_offset = 0
    obj_count = 0

    for stl_path, obj_name in stl_paths_and_names:
        triangles = _stl_to_triangles(stl_path)
        if not triangles:
            continue

        # Deduplicate vertices, round to 4 dp
        vert_map = {}
        verts = []
        tri_indices = []
        for v0, v1, v2 in triangles:
            idxs = []
            for v in (v0, v1, v2):
                key = (round(v[0], 4), round(v[1], 4), round(v[2], 4))
                if key not in vert_map:
                    vert_map[key] = len(verts)
                    verts.append(key)
                idxs.append(vert_map[key])
            tri_indices.append(idxs)

        # OBJ object name: replace whitespace/special chars with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', obj_name)
        lines.append(f'o {safe_name}\n')
        for v in verts:
            lines.append(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
        for idxs in tri_indices:
            # OBJ faces are 1-based and globally indexed
            lines.append(f'f {idxs[0]+vertex_offset+1} {idxs[1]+vertex_offset+1} {idxs[2]+vertex_offset+1}\n')

        vertex_offset += len(verts)
        obj_count += 1

    if obj_count == 0:
        print(f'[WARN] create_obj_bundle: no valid meshes, skipping {output_obj_path}')
        return

    with open(output_obj_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'[INFO] Created OBJ bundle: {output_obj_path} ({obj_count} objects)')


def create_obj_from_items(items, output_obj_path):
    """
    Write a Wavefront OBJ file directly from box dimension data — no temp STL files needed.
    items: list of dicts with keys: name, x, y, z, z_offset (all in mm).
    Each item becomes a named 'o' group = rectangular box (8 verts, 12 triangular faces).
    Works regardless of MESH_STORAGE mode.
    """
    lines = ['# ElevateFoods AI Nutrition Chatbot – multi-part food mesh\n']
    vertex_offset = 0
    obj_count = 0

    for item in items:
        iname  = item['name']
        ix     = float(item['x'])         # width mm
        iy     = float(item['y'])         # depth mm
        iz     = float(item['z'])         # height mm
        z0     = float(item.get('z_offset', 0.0))
        z1     = z0 + iz
        hx, hy = ix / 2.0, iy / 2.0

        verts = [
            (-hx, -hy, z0), ( hx, -hy, z0), ( hx,  hy, z0), (-hx,  hy, z0),
            (-hx, -hy, z1), ( hx, -hy, z1), ( hx,  hy, z1), (-hx,  hy, z1),
        ]
        faces = [
            (0,3,1),(1,3,2),  # bottom
            (4,5,6),(4,6,7),  # top
            (0,4,7),(0,7,3),  # left
            (5,1,2),(5,2,6),  # right
            (0,1,5),(0,5,4),  # front
            (2,3,7),(2,7,6),  # back
        ]

        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', iname)
        lines.append(f'o {safe_name}\n')
        for v in verts:
            lines.append(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n')
        for fi in faces:
            lines.append(f'f {fi[0]+vertex_offset+1} {fi[1]+vertex_offset+1} {fi[2]+vertex_offset+1}\n')
        vertex_offset += len(verts)
        obj_count += 1

    if obj_count == 0:
        print(f'[WARN] create_obj_from_items: no items, skipping {output_obj_path}')
        return

    with open(output_obj_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'[INFO] Written {obj_count} objects to {output_obj_path}')


def recommend(gender, age, height, weight, carbohydrate, protein, fat, activity, diet, preference, preferred_foods=''):
    ensure_nutrition_csv_fresh()

    targets = calculate_macro_targets(gender, age, height, weight, carbohydrate, protein, fat, activity, diet)
    calories = targets['calories']
    carbohydrate_intake = targets['carbohydrate_intake']
    protein_intake = targets['protein_intake']
    fat_intake = targets['fat_intake']
    carbohydrate_needed = targets['carbohydrate_needed']
    protein_needed = targets['protein_needed']
    fat_needed = targets['fat_needed']

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

    def _compute_best_matches(target_need, top_k=4, preferred_foods_text=''):
        """Build 3-4 nutrition-first options from CSV foods (fallback to model foods if needed)."""
        positive_target = np.maximum(target_need, 0.0)
        if np.all(positive_target <= 1e-6):
            return [], {'insufficient': False, 'suggested_foods': []}

        preferred_terms = [
            normalize_food_name(t.strip()) for t in str(preferred_foods_text or '').split(',') if t.strip()
        ]

        def _row_float(row, *keys, default=0.0):
            for key in keys:
                raw = str(row.get(key, '') or '').strip()
                if not raw:
                    continue
                try:
                    return float(raw)
                except Exception:
                    continue
            return default

        def _looks_non_veg(food_name_text):
            n = (food_name_text or '').lower()
            tags = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'mutton', 'lamb', 'meat', 'tuna', 'salmon']
            return any(t in n for t in tags)

        def _food_tags(food_name_text, vec):
            """Heuristic tags for dish composition quality."""
            n = (food_name_text or '').lower()
            carbs_pg, protein_pg, fat_pg = float(vec[0]), float(vec[1]), float(vec[2])

            tags = set()
            if carbs_pg >= max(protein_pg, fat_pg) and carbs_pg > 0.06:
                tags.add('base')
            if protein_pg >= max(carbs_pg, fat_pg) and protein_pg > 0.06:
                tags.add('protein')
            if fat_pg >= max(carbs_pg, protein_pg) and fat_pg > 0.05:
                tags.add('fat')

            veggie_keys = ['broccoli', 'spinach', 'cabbage', 'pepper', 'carrot', 'onion', 'tomato', 'mushroom', 'zucchini', 'lettuce', 'bean', 'pea', 'corn', 'eggplant', 'cauliflower']
            if any(k in n for k in veggie_keys):
                tags.add('veg')

            starch_keys = ['rice', 'noodle', 'pasta', 'potato', 'sweet potato', 'quinoa', 'oat', 'bread']
            if any(k in n for k in starch_keys):
                tags.add('base')

            protein_keys = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'tofu', 'egg', 'lentil', 'bean', 'turkey']
            if any(k in n for k in protein_keys):
                tags.add('protein')

            sauce_fat_keys = ['olive', 'avocado', 'sesame', 'cheese', 'nuts', 'peanut']
            if any(k in n for k in sauce_fat_keys):
                tags.add('fat')

            sweet_keys = ['candy', 'chocolate', 'cake', 'cookie', 'soda', 'syrup']
            if any(k in n for k in sweet_keys):
                tags.add('dessert')

            snack_keys = ['chips', 'popcorn', 'cracker', 'biscuit', 'snack']
            if any(k in n for k in snack_keys):
                tags.add('snack')

            return tags

        def _dish_quality(chosen_items):
            """Score whether selected foods can form one savory dish."""
            union_tags = set()
            names = []
            for it in chosen_items:
                union_tags.update(it.get('tags', set()))
                names.append(it.get('name', ''))

            score = 0.0
            if 'base' in union_tags:
                score += 1.2
            if 'protein' in union_tags:
                score += 1.6
            if 'veg' in union_tags:
                score += 1.0
            if 'fat' in union_tags:
                score += 0.6
            if {'base', 'protein', 'veg'}.issubset(union_tags):
                score += 2.0
            if 'dessert' in union_tags:
                score -= 2.5

            name_blob = ' '.join(n.lower() for n in names)
            if ('rice' in name_blob and ('chicken' in name_blob or 'tofu' in name_blob or 'egg' in name_blob)):
                score += 0.8
            if ('noodle' in name_blob and ('beef' in name_blob or 'chicken' in name_blob or 'tofu' in name_blob)):
                score += 0.8
            if ('potato' in name_blob and ('chicken' in name_blob or 'bean' in name_blob or 'lentil' in name_blob)):
                score += 0.6

            # Build short human-readable dish hint
            if {'base', 'protein', 'veg'}.issubset(union_tags):
                hint = 'Balanced one-bowl meal'
            elif {'protein', 'veg'}.issubset(union_tags):
                hint = 'Savory protein + veggie plate'
            elif {'base', 'protein'}.issubset(union_tags):
                hint = 'Hearty base + protein dish'
            else:
                hint = 'Simple mixed dish'

            return score, hint

        def _pool_from_csv():
            pool = []
            for row in (csv_data or []):
                fname = (row.get('category_name') or '').strip()
                if not fname:
                    continue
                lower_name = fname.lower()
                if normalize_food_name(fname) == 'background':
                    continue
                if preference and _looks_non_veg(fname):
                    continue
                if any(nk in lower_name for nk in ['almond', 'walnut', 'cashew', 'pecan', 'hazelnut', 'pistachio']):
                    continue

                carbs_pg = _row_float(row, 'carbohydrates', 'Carbohydrates', 'carbs', 'Carbs', default=0.0)
                protein_pg = _row_float(row, 'protein', 'Protein', default=0.0)
                fat_pg = _row_float(row, 'fat', 'Fat', default=0.0)
                vec = np.array([carbs_pg, protein_pg, fat_pg], dtype=float)
                if np.sum(vec) <= 1e-8:
                    continue
                tags = _food_tags(fname, vec)
                if 'dessert' in tags or 'snack' in tags:
                    continue
                if not ({'base', 'protein', 'veg', 'fat'} & tags):
                    continue
                norm_name = normalize_food_name(fname)
                pool.append({'name': fname, 'vec': vec, 'tags': tags, 'normalized': norm_name})
            return pool

        def _pool_from_model_foods():
            p = []
            for i in range(len(name)):
                if i == blocked:
                    continue
                p.append({'name': name[i], 'vec': W[i], 'tags': _food_tags(name[i], W[i])})
            return p

        def _is_preferred(item):
            if not preferred_terms:
                return True
            n = item.get('normalized') or normalize_food_name(item.get('name', ''))
            n = n.replace('_', ' ')
            for t in preferred_terms:
                if not t:
                    continue
                tt = t.replace('_', ' ')
                if n == tt or tt in n or n in tt:
                    return True
            return False

        def _select_shortlist(pool, k=18):
            scored = []
            for item in pool:
                probe = item['vec'] * 150.0  # 150g probe serving
                under = np.maximum(positive_target - probe, 0.0)
                over = np.maximum(probe - positive_target, 0.0)
                score = float(np.sum(under) + 3.0 * np.sum(over))
                scored.append((score, item))
            scored.sort(key=lambda x: x[0])
            return [item for _, item in scored[:k]]

        full_pool = _pool_from_csv()
        preferred_pool = [it for it in full_pool if _is_preferred(it)] if preferred_terms else full_pool
        using_preferred = bool(preferred_terms) and len(preferred_pool) >= 2

        pool = preferred_pool if len(preferred_pool) >= 2 else full_pool
        if len(pool) < 6:
            pool = _pool_from_model_foods()
            using_preferred = False

        shortlist = _select_shortlist(pool, k=18 if len(pool) > 18 else len(pool))
        if len(shortlist) < 2:
            return [], {
                'insufficient': bool(preferred_terms),
                'used_preferred': using_preferred,
                'preferred_foods': preferred_terms,
                'reason': 'Not enough preferred foods found in CSV to form combinations.',
                'suggested_foods': [it['name'] for it in _select_shortlist(full_pool, k=5)] if full_pool else [],
            }

        candidates = []
        combo_sizes = [2, 3] if len(shortlist) >= 3 else [2]
        for csize in combo_sizes:
            for idx_tuple in combinations(range(len(shortlist)), csize):
                chosen = [shortlist[i] for i in idx_tuple]
                nutr = np.array([it['vec'] for it in chosen], dtype=float)  # per-gram macros
                try:
                    grams, _ = nnls(nutr.T, positive_target)
                except Exception:
                    continue

                grams = np.clip(grams, 0.0, 350.0)
                if np.sum(grams >= 1.0) < 2:
                    continue

                supplied = np.dot(grams, nutr)
                under = np.maximum(positive_target - supplied, 0.0)
                over = np.maximum(supplied - positive_target, 0.0)
                dish_bonus, dish_hint = _dish_quality(chosen)
                if dish_bonus < 1.4:
                    continue
                score = float(np.sum(under) + 3.0 * np.sum(over) + 0.001 * np.sum(grams) - 0.7 * dish_bonus)

                foods = []
                for j, it in enumerate(chosen):
                    g = round(float(grams[j]), 2)
                    if g >= 1.0:
                        foods.append({'name': it['name'], 'gram': g})
                if len(foods) < 2:
                    continue

                candidates.append({
                    'foods': foods,
                    'dish_hint': dish_hint,
                    'supplied': {
                        'carbs': round(float(supplied[0]), 2),
                        'protein': round(float(supplied[1]), 2),
                        'fat': round(float(supplied[2]), 2),
                    },
                    'shortfall_total': round(float(np.sum(under)), 2),
                    'exceed_total': round(float(np.sum(over)), 2),
                    'dish_score': round(float(dish_bonus), 3),
                    'score': round(score, 3),
                })

        candidates.sort(key=lambda c: (c['score'], c['exceed_total'], c['shortfall_total'], -c.get('dish_score', 0.0)))

        selected = []
        seen_keys = set()
        for c in candidates:
            key = tuple(sorted(f['name'] for f in c['foods']))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(c)
            if len(selected) >= top_k:
                break

        if len(selected) < min(3, top_k):
            for c in candidates:
                if c in selected:
                    continue
                selected.append(c)
                if len(selected) >= top_k:
                    break

        selected = selected[:top_k]

        advice = {
            'insufficient': False,
            'used_preferred': using_preferred,
            'preferred_foods': preferred_terms,
            'suggested_foods': []
        }

        if preferred_terms and not using_preferred:
            advice['insufficient'] = True
            advice['reason'] = 'Preferred foods were not enough to build complete combinations.'
            advice['suggested_foods'] = [it['name'] for it in _select_shortlist(full_pool, k=6)] if full_pool else []
            return selected, advice

        if using_preferred:
            best_gap = selected[0]['shortfall_total'] if selected else float(np.sum(positive_target))
            if (not selected) or best_gap > 18.0:
                advice['insufficient'] = True
                advice['reason'] = 'Preferred foods alone cannot closely meet required nutrition.'

                deficit = positive_target.copy()
                if selected:
                    s0 = selected[0].get('supplied', {})
                    deficit = np.maximum(
                        positive_target - np.array([
                            float(s0.get('carbs', 0) or 0),
                            float(s0.get('protein', 0) or 0),
                            float(s0.get('fat', 0) or 0),
                        ]),
                        0.0
                    )

                extra_pool = [it for it in full_pool if not _is_preferred(it)]
                scored_extra = []
                for it in extra_pool:
                    probe = it['vec'] * 100.0
                    fit = float(np.dot(probe, deficit))
                    if fit > 0:
                        scored_extra.append((fit, it['name']))
                scored_extra.sort(key=lambda x: x[0], reverse=True)
                advice['suggested_foods'] = [n for _, n in scored_extra[:6]]

        return selected, advice

    best_matches, best_match_advice = _compute_best_matches(y, top_k=4, preferred_foods_text=preferred_foods)

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
                material_mesh_list.append({'name': name[indices[i]], 'mesh': mesh_field, 'gram': amounts[i],
                                           'x': round(x, 2), 'y': round(y, 2), 'z': round(z, 2),
                                           'z_offset': round(z_off, 4)})
                cumulative_z += z  # advance the stack by this item's thickness
        # Folder name hint for client-side direct folder save (no ZIP packaging).
        folder_name = ''
        if MESH_MODE != 'none' and material_mesh_list:
            folder_name = f"{datetime.now().strftime('%Y%m%d')}_option{index + 1}"

        results.append((material_mesh_list, round(carbohydrate_supplement, 2), round(protein_supplement, 2), round(fat_supplement, 2), folder_name))

    # print(results)

    recommend_dict = {'calories': round(calories, 2), 
                      'carbohydrate_intake': round(carbohydrate_intake, 2),
                      'protein_intake': round(protein_intake, 2),
                      'fat_intake': round(fat_intake, 2),
                      'carbohydrate_needed': round(carbohydrate_needed, 2),
                      'protein_needed': round(protein_needed, 2),
                      'fat_needed': round(fat_needed, 2),
                                            'best_matches': best_matches,
                                            'best_match_advice': best_match_advice,
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
        preferred_foods = str(user_info.get('preferred_foods', '') or '').strip()

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
            recommend_dict = recommend(gender, age, height, weight, carbs, protein, fat, activity, diet, preference, preferred_foods)
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
                    'best_matches': [],
                    'best_match_advice': {'insufficient': False, 'suggested_foods': []},
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


@app.route('/api/calculate-custom-recipes', methods=['POST'])
def api_calculate_custom_recipes():
    """Calculate Recipe 1-4 from user-entered desired foods."""
    try:
        data = request.json or {}
        user_info = data.get('user_info', {})
        daily_nutrition = data.get('daily_nutrition', {})
        food_text = str(data.get('food_text', '') or '').strip()

        if not food_text:
            return jsonify({'error': 'Please enter foods like "chicken breast, broccoli, noodles".'}), 400

        def to_int(val, default=0):
            try:
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

        gender = to_int(user_info.get('gender'), 0)
        age = to_int(user_info.get('age'), 25)
        height = to_float(user_info.get('height'), 170.0)
        weight = to_float(user_info.get('weight'), 70.0)
        carbs = to_float(daily_nutrition.get('carbs'), 0)
        protein = to_float(daily_nutrition.get('protein'), 0)
        fat = to_float(daily_nutrition.get('fat'), 0)
        activity = to_int(user_info.get('activity'), 2)
        diet = to_int(user_info.get('diet'), 0)
        preference = to_int(user_info.get('preference'), 0)

        targets = calculate_macro_targets(gender, age, height, weight, carbs, protein, fat, activity, diet)
        recipe_data = build_custom_recipe_recommendations(food_text, targets['need_vector'], preference, limit=4)

        return jsonify({
            'success': True,
            'recipes': recipe_data['recipes'],
            'requested_foods': recipe_data['requested_foods'],
            'resolved_foods': recipe_data['resolved_foods'],
            'unresolved_foods': recipe_data['unresolved_foods'],
            'advice': recipe_data['advice'],
        }), 200
    except Exception as e:
        print(f"Error in api_calculate_custom_recipes: {e}")
        if DIAG_MODE:
            return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Custom recipe calculation failed.'}), 500

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

@app.route('/download-obj/<path:filename>', methods=['GET'])
def download_obj(filename):
    """Download a pre-built .obj bundle from local temp storage."""
    try:
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)
        if not os.path.exists(local_path):
            return jsonify({'error': f'OBJ file not found: {filename}'}), 404
        with open(local_path, 'rb') as f:
            file_data = io.BytesIO(f.read())
        file_data.seek(0)
        return send_file(
            file_data,
            mimetype='model/obj',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f'[ERROR] Error downloading OBJ: {e}')
        return jsonify({'error': str(e)}), 404


@app.route('/download-stl/<path:filename>', methods=['GET'])
def download_stl(filename):
    """Download STL file from Google Cloud Storage"""
    try:
        print(f"[DEBUG] Attempting to download STL file: {filename}")
        stl_bytes, err = _load_stl_bytes(filename)
        if err:
            return jsonify({'error': err}), 404

        file_data = io.BytesIO(stl_bytes)
        file_data.seek(0)
        print(f"[DEBUG] Serving STL {filename}, size: {file_data.getbuffer().nbytes} bytes")

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


def _is_safe_stl_filename(filename):
    base = os.path.basename(str(filename or ''))
    return bool(base) and base == filename and base.lower().endswith('.stl')


def _load_stl_bytes(filename):
    """Load one STL as bytes from local/GCS, regenerating local meshes when possible."""
    if not _is_safe_stl_filename(filename):
        return None, f'Invalid STL filename: {filename}'

    if MESH_STORAGE == 'local':
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, filename)
        if not os.path.exists(local_path):
            print(f"[DEBUG] Local STL not found at {local_path}, attempting regeneration from manifest")
            manifest = _load_manifest()
            meta = manifest.get(filename)
            if meta and 'amount' in meta and 'density' in meta:
                try:
                    mesh_generation(
                        filename,
                        float(meta['amount']),
                        float(meta['density']),
                        z_offset=float(meta.get('z_offset', 0.0))
                    )
                except Exception as regen_err:
                    print(f"[WARN] Regeneration failed: {regen_err}")
            if not os.path.exists(local_path):
                return None, f'File not found (local): {filename}'
        with open(local_path, 'rb') as f:
            return f.read(), None

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"meshes/{filename}"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None, f'File not found in storage: {filename}'
    out = io.BytesIO()
    blob.download_to_file(out)
    out.seek(0)
    return out.read(), None


@app.route('/download-stl-zip', methods=['POST'])
def download_stl_zip():
    """Bundle requested STL files into one ZIP for mobile/browser fallback."""
    try:
        payload = request.get_json(silent=True) or {}
        files = payload.get('files', []) or []
        folder_name = str(payload.get('folder_name', 'stl_files') or 'stl_files').strip()

        if not isinstance(files, list) or not files:
            return jsonify({'error': 'No STL files requested.'}), 400

        safe_files = []
        for f in files[:50]:
            fname = os.path.basename(str(f or '').strip())
            if _is_safe_stl_filename(fname):
                safe_files.append(fname)
        if not safe_files:
            return jsonify({'error': 'No valid STL filenames provided.'}), 400

        safe_folder = re.sub(r'[^a-zA-Z0-9_-]+', '_', folder_name).strip('_') or 'stl_files'

        zip_buffer = io.BytesIO()
        missing = []
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in safe_files:
                data, err = _load_stl_bytes(fname)
                if err or data is None:
                    missing.append(fname)
                    continue
                zf.writestr(fname, data)

            if missing:
                zf.writestr('README_missing_files.txt', 'Some files could not be included:\n' + '\n'.join(missing))

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{safe_folder}.zip'
        )
    except Exception as e:
        print(f"[ERROR] Error creating STL ZIP: {e}")
        if DIAG_MODE:
            return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Failed to create STL ZIP.'}), 500

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