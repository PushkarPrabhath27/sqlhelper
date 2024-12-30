from flask import Flask, render_template, request, send_file
import pandas as pd
import io
import csv
import numpy as np
from typing import List, Tuple

app = Flask(__name__)

def detect_delimiter(text: str) -> str:
    """Detect the most likely delimiter in the text data."""
    common_delimiters = [',', '\t', ';', '|', ' ']
    max_splits = 0
    best_delimiter = ','
    
    for delimiter in common_delimiters:
        # Check first few lines
        lines = text.split('\n')[:5]
        splits = [len(line.split(delimiter)) for line in lines if line.strip()]
        if splits and all(s == splits[0] for s in splits) and splits[0] > max_splits:
            max_splits = splits[0]
            best_delimiter = delimiter
    
    return best_delimiter

def clean_column_names(columns: List[str]) -> List[str]:
    """Clean and standardize column names."""
    cleaned = []
    for col in columns:
        # Remove special characters and standardize spacing
        clean = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in str(col))
        clean = '_'.join(clean.split())
        clean = clean.lower()
        
        # Ensure unique names
        base_name = clean
        counter = 1
        while clean in cleaned:
            clean = f"{base_name}_{counter}"
            counter += 1
        
        cleaned.append(clean)
    return cleaned

def infer_data_type(values: List[str]) -> Tuple[str, dict]:
    """Infer the data type and format of a column."""
    # Remove empty values
    non_empty = [v for v in values if str(v).strip()]
    if not non_empty:
        return 'string', {}
    
    # Try numeric
    try:
        # Check if all values are numbers
        numeric_values = pd.to_numeric(non_empty)
        if all(float(n).is_integer() for n in numeric_values):
            return 'integer', {}
        return 'float', {'precision': 2}
    except:
        pass
    
    # Try date
    try:
        pd.to_datetime(non_empty)
        return 'datetime', {'format': 'auto'}
    except:
        pass
    
    # Default to string
    return 'string', {}

def process_raw_data(data: str) -> pd.DataFrame:
    """Process raw input data and convert to DataFrame with intelligent parsing."""
    # Split into lines and remove empty lines
    lines = [line.strip() for line in data.split('\n') if line.strip()]
    if not lines:
        raise ValueError("No data provided")
    
    # Detect delimiter
    delimiter = detect_delimiter(data)
    
    # Parse lines
    rows = [line.split(delimiter) for line in lines]
    
    # Ensure consistent number of columns
    max_cols = max(len(row) for row in rows)
    rows = [row + [''] * (max_cols - len(row)) for row in rows]
    
    # Determine if first row is header
    first_row = rows[0]
    second_row = rows[1] if len(rows) > 1 else []
    
    is_header = True
    if second_row:
        # Check if first row might be data instead of headers
        first_numeric = all(str(x).replace('.','').isdigit() for x in first_row if x.strip())
        if first_numeric:
            is_header = False
    
    if is_header:
        df = pd.DataFrame(rows[1:], columns=first_row)
    else:
        df = pd.DataFrame(rows, columns=[f'Column_{i+1}' for i in range(max_cols)])
    
    # Clean column names
    df.columns = clean_column_names(df.columns)
    
    # Infer and convert data types
    for column in df.columns:
        dtype, params = infer_data_type(df[column].tolist())
        if dtype == 'integer':
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
        elif dtype == 'float':
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
            df[column] = df[column].round(params.get('precision', 2))
        elif dtype == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
    
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return render_template('index.html', error='No file selected')
                
                # Handle different file types
                if file.filename.endswith('.csv'):
                    content = file.read().decode('utf-8')
                    delimiter = detect_delimiter(content)
                    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                elif file.filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    return render_template('index.html', error='Unsupported file format')
                
            elif 'raw_data' in request.form:
                data = request.form['raw_data']
                if not data.strip():
                    return render_template('index.html', error='No data provided')
                
                # Process raw data with intelligent parsing
                df = process_raw_data(data)
            
            # Clean column names
            df.columns = clean_column_names(df.columns)
            
            # Generate HTML table with styling
            table = df.to_html(
                classes='data-table',
                index=False,
                float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
            )
            
            # Prepare CSV data for download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            return render_template('index.html', table=table, csv_data=csv_data)
            
        except Exception as e:
            return render_template('index.html', error=f'Error processing data: {str(e)}')
    
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form['csv_data']
    buffer = io.BytesIO()  # Change to BytesIO
    buffer.write(csv_data.encode('utf-8'))  # Encode as bytes
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='data.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)