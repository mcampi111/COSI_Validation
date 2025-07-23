import pandas as pd
import torch
import numpy as np
from transformers import CamembertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
import sys

# Constants from your training
MAX_LEN = 128
batch_size = 8
device = torch.device('cpu')

# FIXED label mapping (removed duplicates)
label_mapping = {'Autres': 0, 
                 'Dépistage': 1, 
                 'Conversation à 1 ou 2 dans le bruit': 2,
                 'Conversation en groupe dans le silence': 3, 
                 'Conversation à 1 ou 2 dans le silence': 4,
                 'Television/radio au volume normal': 5, 
                 'Conversation en groupe dans le bruit': 6, 
                 'Augmenter le contact social': 7,
                 'Eglise ou reunion': 8, 
                 'Se sentir embarrasse ou stupide': 9,
                'Interlocuteur non familier au telephone': 10,
                'Interlocuteur familier au telephone': 11,
                 "Entendre la sonnette de la porte ou quelqu'un frapper": 12, 
                 'Entendre le trafic': 13,
                "Entendre le telephone sonner d'une autre piece": 14,
                'Se sentir exclu': 15}

# Reverse mapping for converting predictions back to labels
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def classify_texts(texts, model, tokenizer):
    """Classify a list of texts using the trained model"""
    if not texts or len(texts) == 0:
        return []
    
    # Remove NaN values and convert to string
    clean_texts = [str(text) if pd.notna(text) else "" for text in texts]
    
    # Tokenize
    tokenized_ids = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN) for text in clean_texts]
    tokenized_ids = pad_sequences(tokenized_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Create attention masks
    attention_masks = [[float(i>0) for i in seq] for seq in tokenized_ids]
    
    # Create DataLoader for batching
    data = torch.utils.data.TensorDataset(torch.tensor(tokenized_ids), torch.tensor(attention_masks))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    
    predictions = []
    total_batches = len(dataloader)
    
    print(f"Processing {len(clean_texts)} texts in {total_batches} batches...")
    sys.stdout.flush()
    
    # Apply model in batches
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            
            predictions.extend(np.argmax(logits, axis=1).flatten())
            
            # Progress logging every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
                sys.stdout.flush()
    
    # Convert predictions to labels with error handling
    predicted_labels = []
    for pred in predictions:
        if pred in reverse_label_mapping:
            predicted_labels.append(reverse_label_mapping[pred])
        else:
            print(f"Warning: Unknown prediction {pred}, using 'Autres'")
            sys.stdout.flush()
            predicted_labels.append('Autres')  # Default fallback
    
    return predicted_labels

def main():
    # Set cache locations first to avoid disk space issues
    import os
    os.environ['HF_HOME'] = '/pasteur/appa/scratch/mcampi/COSI2/.cache'
    os.environ['TRANSFORMERS_CACHE'] = '/pasteur/appa/scratch/mcampi/COSI2/.cache'
    
    print("=== STARTING MAIN ===")
    sys.stdout.flush()
    
    print("Loading model and tokenizer...")
    sys.stdout.flush()
    
    # Load tokenizer first
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    print("Tokenizer loaded successfully")
    sys.stdout.flush()
    
    # Try to load saved model with error handling
    try:
        # Load your trained model with the new method
        model = torch.load('data/working_trained_model.pth', map_location=device, weights_only=False)
        print("SUCCESS: Loaded your trained model!")
        sys.stdout.flush()
        model_type = "trained"
    except Exception as e:
        print(f"ERROR loading trained model: {e}")
        sys.stdout.flush()
        print("Loading base CamemBERT model instead...")
        sys.stdout.flush()
        from transformers import CamembertForSequenceClassification
        model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=17)  # Fixed: 17 labels
        print("WARNING: Using untrained base model - results will be random!")
        sys.stdout.flush()
        model_type = "base"
    
    print("Moving model to device...")
    sys.stdout.flush()
    model.to(device)
    print("Model ready!")
    sys.stdout.flush()
    
    print("Loading Excel file...")
    sys.stdout.flush()
    
    # Load Excel file
    excel_file_path = 'data/COSI_EVOLUTION_by_need.xlsx'
    
    # Sheets to process (only need1 for now)
    sheets_to_process = ['need1']
    
    # Dictionary to store all sheets (including unprocessed ones)
    all_sheets = {}
    
    # First, load all sheets to preserve the original structure
    with pd.ExcelFile(excel_file_path, engine='openpyxl') as xls:
        for sheet_name in xls.sheet_names:
            all_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
    
    print(f"Available sheets: {list(all_sheets.keys())}")
    sys.stdout.flush()
    
    # Process the specific sheets
    for sheet_name in sheets_to_process:
        if sheet_name in all_sheets:
            print(f"\nProcessing {sheet_name}...")
            sys.stdout.flush()
            df = all_sheets[sheet_name]
            
            print(f"Shape: {df.shape}")
            sys.stdout.flush()
            print(f"Columns: {df.columns.tolist()}")
            sys.stdout.flush()
            
            if 'OPEN_ANSWER' in df.columns:
                # Get texts for classification
                texts = df['OPEN_ANSWER'].tolist()
                print(f"Found {len(texts)} texts to classify")
                sys.stdout.flush()
                
                # Classify
                print("Starting classification...")
                sys.stdout.flush()
                predicted_labels = classify_texts(texts, model, tokenizer)
                print("Classification completed!")
                sys.stdout.flush()
                
                # Add Label column
                df['Label'] = predicted_labels
                all_sheets[sheet_name] = df
                
                print(f"Added {len(predicted_labels)} predictions to {sheet_name}")
                sys.stdout.flush()
                
                # Show some examples
                print("\nSample predictions:")
                sys.stdout.flush()
                for i in range(min(3, len(texts))):
                    if pd.notna(texts[i]) and texts[i].strip():
                        print(f"Text: {texts[i][:100]}...")
                        print(f"Label: {predicted_labels[i]}")
                        print("---")
                        sys.stdout.flush()
            else:
                print(f"WARNING: 'OPEN_ANSWER' column not found in {sheet_name}")
                sys.stdout.flush()
                print(f"Available columns: {df.columns.tolist()}")
                sys.stdout.flush()
        else:
            print(f"WARNING: Sheet '{sheet_name}' not found in Excel file")
            sys.stdout.flush()
    
    # Save results
    output_filename = f'COSI_EVOLUTION_by_need_labeled_{model_type}.xlsx'
    output_path = f'results/{output_filename}'
    print(f"\nSaving results to {output_path}...")
    sys.stdout.flush()
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Classification completed successfully using {model_type} model!")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
