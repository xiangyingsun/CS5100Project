# 2. data_preprocessor.py
import pandas as pd
import logging
from typing import Optional

def preprocess_data(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Processes yoga poses CSV into structured format for embedding
    Returns DataFrame and saves cleaned data if output_path specified
    """
    try:
        df = pd.read_csv(input_path, keep_default_na=False)
        
        # Create combined text content
        df['content'] = df.apply(lambda x: 
            f"Pose Name: {x['AName']}\n"
            f"Description: {x['Description']}\n"
            f"Benefits: {x['Benefits']}\n"
            f"Breathing: {x['Breathing']}\n"
            f"Awareness: {x['Awareness']}", axis=1)
            
        # Clean metadata
        meta_cols = ['AID', 'Level', 'Contraindications', 'Variations']
        df_meta = df[meta_cols].copy()
        df_meta = df_meta.where(pd.notnull(df_meta), None)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Cleaned data saved to {output_path}")
            
        return df_meta, df['content'].tolist()
    
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        raise
