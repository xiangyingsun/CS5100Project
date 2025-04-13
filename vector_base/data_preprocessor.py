# 2. data_preprocessor.py
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict
import re

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if pd.isna(text):
        return None
    
    # Convert numeric values to strings
    if not isinstance(text, str):
        text = str(text)
        
    # Check if empty after stripping
    if text.strip() == '':
        return None
        
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(input_path: str, output_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Processes yoga poses CSV into structured format for embedding
    Returns DataFrame and saves cleaned data if output_path specified
    """
    try:
        # Read CSV with proper handling of multi-line fields
        df = pd.read_csv(input_path, keep_default_na=False, encoding='utf-8')
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Standardize column names
        df = df.rename(columns={'awareness': 'Awareness'})
        
        # Clean text content in all text columns
        text_columns = ['Description', 'Benefits', 'Breathing', 'Awareness']
        for col in text_columns:
            df[col] = df[col].apply(clean_text)
        
        # Create combined text content with fallbacks for empty fields
        df['content'] = df.apply(lambda x: 
            f"Pose Name: {x['AName']}\n"
            f"Description: {x['Description'] or 'No description available'}\n"
            f"Benefits: {x['Benefits'] or 'No benefits listed'}\n"
            f"Breathing: {x['Breathing'] or 'Normal breathing'}\n"
            f"Awareness: {x['Awareness'] or 'General body awareness'}", axis=1)
            
        # Clean metadata
        meta_cols = ['AID', 'Level', 'Contraindications', 'Variations']
        df_meta = df[meta_cols].copy()
        
        # Clean metadata values
        for col in meta_cols:
            df_meta[col] = df_meta[col].apply(clean_text)
            
        # Replace None values with empty strings for ChromaDB compatibility
        df_meta = df_meta.fillna('')
        
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Cleaned data saved to {output_path}")
            
        return df_meta, df['content'].tolist()
    
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        raise
