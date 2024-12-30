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
        clean = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in str(col))
        clean = '_'.join(clean.split())
        clean = clean.lower()
        
        base_name = clean
        counter = 1
        while clean in cleaned:
            clean = f"{base_name}_{counter}"
            counter += 1
        
        cleaned.append(clean)
    return cleaned

def infer_data_type(values: List[str]) -> Tuple[str, dict]:
    """Infer the data type and format of a column."""
    non_empty = [v for v in values if str(v).strip()]
    if not non_empty:
        return 'string', {}
    
    try:
        numeric_values = pd.to_numeric(non_empty)
        if all(float(n).is_integer() for n in numeric_values):
            return 'integer', {}
        return 'float', {'precision': 2}
    except:
        pass
    
    try:
        pd.to_datetime(non_empty)
        return 'datetime', {'format': 'auto'}
    except:
        pass
    
    return 'string', {}

def process_raw_data(data: str) -> pd.DataFrame:
    """Process raw input data and convert to DataFrame with intelligent parsing."""
    lines = [line.strip() for line in data.split('\n') if line.strip()]
    if not lines:
        raise ValueError("No data provided")
    
    delimiter = detect_delimiter(data)
    rows = [line.split(delimiter) for line in lines]
    
    max_cols = max(len(row) for row in rows)
    rows = [row + [''] * (max_cols - len(row)) for row in rows]
    
    first_row = rows[0]
    second_row = rows[1] if len(rows) > 1 else []
    
    is_header = True
    if second_row:
        first_numeric = all(str(x).replace('.','').isdigit() for x in first_row if x.strip())
        if first_numeric:
            is_header = False
    
    if is_header:
        df = pd.DataFrame(rows[1:], columns=first_row)
    else:
        df = pd.DataFrame(rows, columns=[f'Column_{i+1}' for i in range(max_cols)])
    
    df.columns = clean_column_names(df.columns)
    
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
                
                df = process_raw_data(data)
            
            df.columns = clean_column_names(df.columns)
            
            # Generate HTML table
            table = df.to_html(
                classes='data-table',
                index=False,
                float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
            )
            
            # Store DataFrame in session
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
    buffer = io.BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='data.csv'
    )

if __name__ == '__main__':
    app.run(debug=True) 