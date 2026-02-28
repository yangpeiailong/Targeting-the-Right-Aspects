import openpyxl
import os
import re
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import jieba
import warnings
import numpy as np
from pathlib import Path

# Ignore irrelevant warning messages
warnings.filterwarnings('ignore')

# Set matplotlib backend to avoid PyCharm compatibility issues
plt.switch_backend('TkAgg')

# -------------------------- Configuration Parameters --------------------------
DATASET_NAME = '人工智能1'  # Keep Chinese for file path matching
DATA_ROOT_PATH = str(Path(__file__).parent.parent.parent.absolute())
excel_file_path = os.path.join(DATA_ROOT_PATH, f"data/reviews_after_annotation/{DATASET_NAME}_attributes_for_each_review2.xlsx")
stopword_file_path = os.path.join(DATA_ROOT_PATH, f"data/stop_words/stop_words.txt")
min_topic_number = 1
max_topic_number = 30
random_seed = 42

# -----------------------------------------------------------------------------

def filter_special_characters(text):
    """
    Filter special characters, non-Chinese characters, numbers and single-character symbols
    Args:
        text (str): Original text string
    Returns:
        str: Cleaned text with only Chinese characters
    """
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def load_and_preprocess_text_data():
    """
    Load and preprocess text data from Excel file with special character filtering
    Returns:
        texts (list): List of tokenized texts after stopword removal and special char filtering
        dictionary (gensim.corpora.Dictionary): Gensim dictionary object
        corpus (list): Bag-of-words corpus
        corpus_tfidf (list): TF-IDF weighted corpus
    """
    workbook = openpyxl.load_workbook(excel_file_path, read_only=True)
    worksheet = workbook.active

    raw_texts = []
    for row_index, row in enumerate(worksheet.iter_rows(values_only=True)):
        if row_index == 0:
            continue
        first_column_content = row[0]
        if not first_column_content or not isinstance(first_column_content, str):
            continue

        cleaned_text = filter_special_characters(first_column_content)
        tokenized_text = [token.strip() for token in jieba.lcut(cleaned_text) if token.strip()]
        raw_texts.append(tokenized_text)

    # Load stopwords with encoding compatibility
    try:
        with open(stopword_file_path, 'r', encoding='gbk') as f:
            stopwords = [word.strip() for word in f.readlines()]
    except UnicodeDecodeError:
        with open(stopword_file_path, 'r', encoding='utf-8') as f:
            stopwords = [word.strip() for word in f.readlines()]

    # Filter stopwords, empty strings and non-Chinese single characters
    processed_texts = []
    for text in raw_texts:
        filtered_tokens = []
        for token in text:
            if (token not in stopwords and
                token and
                len(token) > 1 and
                re.match(r'^[\u4e00-\u9fa5]+$', token)):
                filtered_tokens.append(token)
        if filtered_tokens:
            processed_texts.append(filtered_tokens)

    # Create dictionary and corpus with extreme frequency filtering
    text_dictionary = corpora.Dictionary(processed_texts)
    text_dictionary.filter_extremes(no_below=2, no_above=0.9)
    bow_corpus = [text_dictionary.doc2bow(text) for text in processed_texts]

    # Convert to TF-IDF representation
    tfidf_transformer = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_transformer[bow_corpus]

    print(f"Preprocessing completed: {len(processed_texts)} valid texts, {len(text_dictionary)} unique tokens")
    return processed_texts, text_dictionary, bow_corpus, tfidf_corpus

def calculate_lda_evaluation_metrics(texts, dictionary, bow_corpus, tfidf_corpus):
    """
    Calculate perplexity and coherence scores for different topic numbers
    Args:
        texts (list): Preprocessed tokenized texts
        dictionary (gensim.corpora.Dictionary): Gensim dictionary
        bow_corpus (list): Bag-of-words corpus
        tfidf_corpus (list): TF-IDF corpus
    Returns:
        topic_numbers (list): List of topic numbers tested
        perplexity_scores (list): Log perplexity scores for each topic number
        coherence_scores (list): C_V coherence scores for each topic number
    """
    topic_numbers = []
    perplexity_scores = []
    coherence_scores = []

    print(f"\nCalculating metrics for {min_topic_number}-{max_topic_number} topics...")
    for num_topics in tqdm(range(min_topic_number, max_topic_number + 1)):
        lda_model = models.LdaModel(
            corpus=tfidf_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=random_seed,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )

        # Calculate evaluation metrics
        perplexity = lda_model.log_perplexity(tfidf_corpus)
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        # Store results
        topic_numbers.append(num_topics)
        perplexity_scores.append(perplexity)
        coherence_scores.append(coherence_score)

    return topic_numbers, perplexity_scores, coherence_scores

def plot_evaluation_metrics(topic_numbers, perplexity_scores, coherence_scores):
    """
    Plot perplexity and coherence scores against number of topics
    Args:
        topic_numbers (list): List of topic numbers
        perplexity_scores (list): Log perplexity scores
        coherence_scores (list): C_V coherence scores
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot perplexity (left axis)
    color_perplexity = 'tab:red'
    ax1.set_xlabel('Number of Topics', fontsize=12)
    ax1.set_ylabel('Log Perplexity', color=color_perplexity, fontsize=12)
    ax1.plot(topic_numbers, perplexity_scores, color=color_perplexity,
             linewidth=2, marker='o', markersize=4, label='Perplexity')
    ax1.tick_params(axis='y', labelcolor=color_perplexity)
    ax1.grid(True, alpha=0.3)

    # Plot coherence (right axis)
    ax2 = ax1.twinx()
    color_coherence = 'tab:blue'
    ax2.set_ylabel('Coherence (c_v)', color=color_coherence, fontsize=12)
    ax2.plot(topic_numbers, coherence_scores, color=color_coherence,
             linewidth=2, marker='s', markersize=4, label='Coherence')
    ax2.tick_params(axis='y', labelcolor=color_coherence)

    # Set x-axis ticks and plot properties
    ax1.set_xticks(range(min_topic_number, max_topic_number + 1, 2))
    fig.suptitle('LDA Topic Number Selection: Perplexity vs Coherence', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Adjust layout and save plot
    fig.tight_layout()
    plot_save_path = os.path.join(DATA_ROOT_PATH, f"data/results_for_unsupervised_methods/{DATA_ROOT_PATH}_Perplexity_Coherence_Topics.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def print_optimal_topic_number_recommendations(topic_numbers, perplexity_scores, coherence_scores):
    """
    Print recommendations for optimal number of topics based on metrics
    Args:
        topic_numbers (list): List of topic numbers
        perplexity_scores (list): Log perplexity scores
        coherence_scores (list): C_V coherence scores
    """
    # Find optimal topic numbers
    min_perplexity_index = np.argmin(perplexity_scores)
    optimal_perplexity_topic = topic_numbers[min_perplexity_index]
    min_perplexity_value = perplexity_scores[min_perplexity_index]

    max_coherence_index = np.argmax(coherence_scores)
    optimal_coherence_topic = topic_numbers[max_coherence_index]
    max_coherence_value = coherence_scores[max_coherence_index]

    # Print results
    print("\n==================== Metrics Summary ====================")
    print(f"Topic number with minimum perplexity: {optimal_perplexity_topic} (Value: {min_perplexity_value:.4f})")
    print(f"Topic number with maximum coherence: {optimal_coherence_topic} (Value: {max_coherence_value:.4f})")
    print("==========================================================")
    print("\nRecommendations:")
    print(f"1. Prioritize {optimal_coherence_topic} topics (highest coherence, better topic interpretability)")
    print(f"2. Secondary option: {optimal_perplexity_topic} topics (lowest perplexity, better model fit)")
    print("3. Final selection requires manual inspection of topic keywords for practical interpretability")

if __name__ == '__main__':
    processed_texts, text_dict, bow_corpus, tfidf_corpus = load_and_preprocess_text_data()
    topic_nums, perplexity_vals, coherence_vals = calculate_lda_evaluation_metrics(
        processed_texts, text_dict, bow_corpus, tfidf_corpus
    )
    plot_evaluation_metrics(topic_nums, perplexity_vals, coherence_vals)
    print_optimal_topic_number_recommendations(topic_nums, perplexity_vals, coherence_vals)