# BookCorpus Story Generation Pipeline - Complete Guide

This notebook provides a step-by-step pipeline to load the BookCorpus dataset, explore it statistically, and fine-tune transformer models for story generation.

## Prerequisites & Setup

### Cell 1: Install Required Libraries
```python
# Install required packages
!pip install transformers datasets torch accelerate evaluate
!pip install matplotlib seaborn wordcloud
!pip install nltk textstat
```

### Cell 2: Import Libraries
```python
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
import textstat
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### Cell 3: Check GPU Availability
```python
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## Part 1: Dataset Loading and Exploration

### Cell 4: Load BookCorpus Dataset
```python
# Load BookCorpus dataset
# Note: BookCorpus is large, so we'll work with a subset for demonstration
print("Loading BookCorpus dataset...")

try:
    # Load the dataset - using a subset for memory efficiency
    dataset = load_dataset("bookcorpus", split="train[:10000]")  # First 10k samples
    print(f"Dataset loaded successfully!")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using a smaller alternative dataset for demonstration...")
    # Fallback to a smaller story dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")

# Convert to pandas for easier exploration
df = dataset.to_pandas()
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

### Cell 5: Basic Dataset Statistics
```python
# Basic statistics about the dataset
print("=== BASIC DATASET STATISTICS ===")
print(f"Total number of texts: {len(df)}")
print(f"Dataset memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Text column analysis (assuming 'text' column exists)
text_column = 'text' if 'text' in df.columns else df.columns[0]
texts = df[text_column].tolist()

# Calculate text lengths
text_lengths = [len(text) for text in texts]
word_counts = [len(text.split()) for text in texts]

print(f"\n=== TEXT LENGTH STATISTICS ===")
print(f"Average text length (characters): {np.mean(text_lengths):.2f}")
print(f"Median text length (characters): {np.median(text_lengths):.2f}")
print(f"Min text length: {min(text_lengths)}")
print(f"Max text length: {max(text_lengths)}")

print(f"\n=== WORD COUNT STATISTICS ===")
print(f"Average word count: {np.mean(word_counts):.2f}")
print(f"Median word count: {np.median(word_counts):.2f}")
print(f"Min word count: {min(word_counts)}")
print(f"Max word count: {max(word_counts)}")
```

### Cell 6: Text Length Distribution Visualization
```python
# Visualize text length distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Character length distribution
axes[0,0].hist(text_lengths, bins=50, alpha=0.7, color='blue')
axes[0,0].set_title('Distribution of Text Lengths (Characters)')
axes[0,0].set_xlabel('Number of Characters')
axes[0,0].set_ylabel('Frequency')

# Word count distribution
axes[0,1].hist(word_counts, bins=50, alpha=0.7, color='green')
axes[0,1].set_title('Distribution of Word Counts')
axes[0,1].set_xlabel('Number of Words')
axes[0,1].set_ylabel('Frequency')

# Box plot for character lengths
axes[1,0].boxplot(text_lengths)
axes[1,0].set_title('Box Plot: Text Lengths (Characters)')
axes[1,0].set_ylabel('Number of Characters')

# Box plot for word counts
axes[1,1].boxplot(word_counts)
axes[1,1].set_title('Box Plot: Word Counts')
axes[1,1].set_ylabel('Number of Words')

plt.tight_layout()
plt.show()

# Print percentiles
print("Character Length Percentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"{p}th percentile: {np.percentile(text_lengths, p):.0f} characters")
```

### Cell 7: Vocabulary Analysis
```python
# Vocabulary analysis
print("=== VOCABULARY ANALYSIS ===")

# Combine all text and tokenize
all_text = " ".join(texts)
words = re.findall(r'\b\w+\b', all_text.lower())

# Calculate vocabulary statistics
unique_words = set(words)
word_freq = Counter(words)

print(f"Total words: {len(words):,}")
print(f"Unique words (vocabulary size): {len(unique_words):,}")
print(f"Vocabulary richness (TTR): {len(unique_words)/len(words):.4f}")

# Most common words
print(f"\n=== TOP 20 MOST FREQUENT WORDS ===")
for word, count in word_freq.most_common(20):
    print(f"{word:15s}: {count:,} ({count/len(words)*100:.2f}%)")

# Rare words analysis
singleton_words = sum(1 for count in word_freq.values() if count == 1)
print(f"\nWords appearing only once: {singleton_words:,} ({singleton_words/len(unique_words)*100:.2f}%)")
```

### Cell 8: Reading Complexity Analysis
```python
# Reading complexity analysis using textstat
print("=== READING COMPLEXITY ANALYSIS ===")

# Sample a subset for complexity analysis (it's computationally expensive)
sample_texts = texts[:100] if len(texts) > 100 else texts

flesch_scores = []
fk_grades = []
sentence_counts = []

for text in sample_texts:
    # Clean text for better analysis
    clean_text = re.sub(r'[^\w\s\.\!\?]', '', text)
    if len(clean_text.strip()) > 10:  # Only analyze non-empty texts
        flesch_scores.append(textstat.flesch_reading_ease(clean_text))
        fk_grades.append(textstat.flesch_kincaid_grade(clean_text))
        sentence_counts.append(textstat.sentence_count(clean_text))

if flesch_scores:
    print(f"Average Flesch Reading Ease: {np.mean(flesch_scores):.2f}")
    print(f"Average Flesch-Kincaid Grade: {np.mean(fk_grades):.2f}")
    print(f"Average sentences per text: {np.mean(sentence_counts):.2f}")
    
    # Interpretation
    avg_flesch = np.mean(flesch_scores)
    if avg_flesch >= 90:
        difficulty = "Very Easy"
    elif avg_flesch >= 80:
        difficulty = "Easy"
    elif avg_flesch >= 70:
        difficulty = "Fairly Easy"
    elif avg_flesch >= 60:
        difficulty = "Standard"
    elif avg_flesch >= 50:
        difficulty = "Fairly Difficult"
    elif avg_flesch >= 30:
        difficulty = "Difficult"
    else:
        difficulty = "Very Difficult"
    
    print(f"Reading Difficulty Level: {difficulty}")
```

### Cell 9: Word Cloud Visualization
```python
# Create word cloud
print("Generating word cloud...")

# Remove common stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Filter out stop words and short words
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
filtered_text = " ".join(filtered_words)

# Generate word cloud
plt.figure(figsize=(12, 8))
wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    max_words=100,
    colormap='viridis'
).generate(filtered_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in BookCorpus (excluding stop words)', fontsize=16)
plt.tight_layout()
plt.show()
```

### Cell 10: Sample Text Analysis
```python
# Display sample texts with analysis
print("=== SAMPLE TEXTS ANALYSIS ===")

for i, text in enumerate(texts[:3]):  # Show first 3 texts
    print(f"\n--- SAMPLE {i+1} ---")
    print(f"Length: {len(text)} characters, {len(text.split())} words")
    print(f"First 200 characters: {text[:200]}...")
    
    # Count sentences
    sentence_count = len(re.findall(r'[.!?]+', text))
    print(f"Estimated sentences: {sentence_count}")
    
    # Find most common words in this text
    text_words = re.findall(r'\b\w+\b', text.lower())
    text_word_freq = Counter(text_words)
    print(f"Most common words: {dict(text_word_freq.most_common(5))}")
```

## Part 2: Data Preprocessing for Model Training

### Cell 11: Text Preprocessing Function
```python
def preprocess_text(text):
    """
    Preprocess text for story generation model training
    """
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    # Ensure text ends with proper punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

# Apply preprocessing
print("Preprocessing texts...")
processed_texts = [preprocess_text(text) for text in texts]

# Filter out very short texts (less than 50 characters)
processed_texts = [text for text in processed_texts if len(text) >= 50]

print(f"Texts after preprocessing: {len(processed_texts)}")
print(f"Sample processed text: {processed_texts[0][:200]}...")
```

### Cell 12: Model and Tokenizer Setup
```python
# Choose a lightweight model for fine-tuning
MODEL_NAME = "gpt2"  # You can also use "distilgpt2" for even lighter model

print(f"Loading model and tokenizer: {MODEL_NAME}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded: {model.config.n_positions} max positions")
print(f"Vocab size: {tokenizer.vocab_size}")
```

### Cell 13: Tokenization and Dataset Preparation
```python
# Tokenization parameters
MAX_LENGTH = 512  # Adjust based on your model and GPU memory

def tokenize_function(examples):
    """Tokenize texts for causal language modeling"""
    # Tokenize with truncation and padding
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Create dataset from processed texts
from datasets import Dataset

# Create train/validation split
split_idx = int(0.9 * len(processed_texts))
train_texts = processed_texts[:split_idx]
val_texts = processed_texts[split_idx:]

# Create datasets
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Tokenize datasets
print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Tokenization complete!")
```

### Cell 14: Data Collator Setup
```python
# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
    return_tensors="pt"
)

print("Data collator configured for causal language modeling")
```

## Part 3: Model Fine-tuning

### Cell 15: Training Arguments
```python
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./story-generation-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    warmup_steps=100,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
    report_to=None,  # Disable wandb/tensorboard
    dataloader_pin_memory=False,
)

print("Training arguments configured:")
print(f"- Epochs: {training_args.num_train_epochs}")
print(f"- Batch size: {training_args.per_device_train_batch_size}")
print(f"- Learning rate: {training_args.learning_rate}")
print(f"- Warmup steps: {training_args.warmup_steps}")
```

### Cell 16: Initialize Trainer
```python
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("Trainer initialized successfully!")
print(f"Total training steps: {len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
```

### Cell 17: Start Training
```python
# Start training
print("Starting model training...")
print("This may take a while depending on your dataset size and hardware...")

# Train the model
training_result = trainer.train()

print("Training completed!")
print(f"Final training loss: {training_result.training_loss:.4f}")

# Save the fine-tuned model
trainer.save_model("./story-generation-final")
tokenizer.save_pretrained("./story-generation-final")
print("Model saved successfully!")
```

## Part 4: Model Evaluation and Testing

### Cell 18: Generate Sample Stories
```python
# Load the fine-tuned model for inference
print("Loading fine-tuned model for story generation...")

# Move model to appropriate device
model = model.to(device)
model.eval()

def generate_story(prompt, max_length=200, temperature=0.8, num_return_sequences=1):
    """
    Generate story continuation from a prompt
    """
    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
    # Decode and return generated text
    generated_texts = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

# Test story generation with sample prompts
test_prompts = [
    "Once upon a time, in a magical forest,",
    "The old house on the hill had been abandoned for",
    "Sarah opened the mysterious letter and discovered",
    "In the year 2050, humanity made contact with"
]

print("=== GENERATED STORIES ===")
for i, prompt in enumerate(test_prompts):
    print(f"\n--- Story {i+1} ---")
    print(f"Prompt: {prompt}")
    print("Generated continuation:")
    
    generated = generate_story(prompt, max_length=150)
    print(generated[0])
    print("-" * 50)
```

### Cell 19: Interactive Story Generation
```python
# Interactive story generation function
def interactive_story_generation():
    """
    Interactive function for story generation
    """
    print("=== INTERACTIVE STORY GENERATOR ===")
    print("Enter a story prompt and I'll continue it!")
    print("Type 'quit' to exit")
    
    while True:
        user_prompt = input("\nEnter your story prompt: ")
        
        if user_prompt.lower() == 'quit':
            print("Thanks for using the story generator!")
            break
            
        if len(user_prompt.strip()) < 5:
            print("Please provide a longer prompt (at least 5 characters)")
            continue
            
        try:
            # Generate multiple variations
            generated_stories = generate_story(
                user_prompt, 
                max_length=200, 
                temperature=0.8, 
                num_return_sequences=2
            )
            
            print(f"\n--- Generated Story Variations ---")
            for i, story in enumerate(generated_stories):
                print(f"\nVariation {i+1}:")
                print(story)
                print("-" * 40)
                
        except Exception as e:
            print(f"Error generating story: {e}")

# Note: In Colab, you can run this function to test interactively
# interactive_story_generation()

print("Interactive story generation function ready!")
print("Uncomment the last line to run it interactively.")
```

### Cell 20: Model Evaluation Metrics
```python
# Evaluate the model on validation set
print("Evaluating model performance...")

eval_results = trainer.evaluate()

print("=== MODEL EVALUATION RESULTS ===")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Calculate perplexity
perplexity = np.exp(eval_results['eval_loss'])
print(f"\nPerplexity: {perplexity:.2f}")

# Perplexity interpretation
if perplexity < 50:
    quality = "Excellent"
elif perplexity < 100:
    quality = "Good"
elif perplexity < 200:
    quality = "Fair"
else:
    quality = "Needs Improvement"

print(f"Model Quality: {quality}")
```

### Cell 21: Save and Export Results
```python
# Save training history and model information
import json

# Create model info dictionary
model_info = {
    "model_name": MODEL_NAME,
    "training_texts": len(processed_texts),
    "vocab_size": tokenizer.vocab_size,
    "max_length": MAX_LENGTH,
    "training_epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "final_eval_loss": eval_results['eval_loss'],
    "perplexity": float(perplexity),
    "model_quality": quality
}

# Save model info
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print("=== TRAINING SUMMARY ===")
print(f"Base Model: {MODEL_NAME}")
print(f"Training Samples: {len(processed_texts):,}")
print(f"Final Evaluation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"Model Quality: {quality}")
print("\nModel and training info saved!")

# Display final instructions
print("\n=== NEXT STEPS ===")
print("1. Test your model with different prompts")
print("2. Adjust training parameters if needed (epochs, batch size, learning rate)")
print("3. Try different base models (distilgpt2, gpt2-medium)")
print("4. Experiment with generation parameters (temperature, top_k, top_p)")
print("5. Consider using more training data for better results")
```

## Usage Tips and Best Practices

### Model Selection Guide:
- **distilgpt2**: Fastest, good for experimentation
- **gpt2**: Balanced performance and speed
- **gpt2-medium**: Better quality, requires more memory
- **gpt2-large**: Best quality, requires significant GPU memory

### Training Tips:
1. **Start Small**: Begin with a subset of data to test your pipeline
2. **Monitor Loss**: Training loss should decrease steadily
3. **Avoid Overfitting**: Use validation loss to determine when to stop
4. **Experiment**: Try different hyperparameters (learning rate, batch size)

### Generation Tips:
1. **Temperature**: Lower (0.5-0.7) for more focused text, higher (0.8-1.2) for creativity
2. **Top-k/Top-p**: Use top_k=50 and top_p=0.95 for balanced results
3. **Prompt Quality**: Good prompts lead to better continuations
4. **Length Control**: Adjust max_length based on desired story length

### Troubleshooting:
- **Out of Memory**: Reduce batch_size or max_length
- **Poor Generation**: Try more training epochs or better data preprocessing
- **Repetitive Text**: Adjust temperature and top_k/top_p parameters
