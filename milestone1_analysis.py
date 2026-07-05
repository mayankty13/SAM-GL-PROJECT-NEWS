import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text as sklearn_text
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'milestone1_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILES = [
    DATA_DIR / 'business_data.csv',
    DATA_DIR / 'education_data.csv',
    DATA_DIR / 'entertainment_data.csv',
    DATA_DIR / 'sports_data.csv',
    DATA_DIR / 'technology_data.csv',
]

STOP_WORDS = sklearn_text.ENGLISH_STOP_WORDS

# Choose one vectorization method by uncommenting the one you want to use.
VECTORIZATION_METHOD = 'tfidf'  # options: 'bow', 'tfidf', 'embeddings'
DROP_OUTLIERS = False
MIN_CONTENT_WORDS = 5
MAX_CONTENT_WORDS = 5000
MAX_FEATURES = 10000
EMBEDDING_SIZE = 100
REDUNDANT_SENTENCE_MIN_FREQUENCY = 3
REDUNDANT_SENTENCE_MIN_TOKENS = 3
REDUNDANT_SENTENCE_MAX_TOKENS = 25
REDUNDANT_SENTENCE_MIN_UNIQUE_RATIO = 0.25
AUTO_REMOVE_REDUNDANT_SENTENCES = True


def load_data(files):
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df['source_file'] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def print_section(title):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80 + '\n')


def data_report(df):
    print_section('Milestone 1 - Data Report')
    print('Total rows:', len(df))
    print('Total columns:', len(df.columns))
    print('Columns:', df.columns.tolist())
    print('\nColumn types:')
    print(df.dtypes)
    print('\nMissing values by column:')
    print(df.isna().sum())
    print('\nNumber of duplicate rows:', df.duplicated().sum())
    print('\nCategory distribution:')
    print(df['category'].value_counts())
    print('\nSample rows:')
    print(df.head(3).to_dict(orient='records'))


def compute_length_features(df):
    df['headline_length'] = df['headlines'].fillna('').astype(str).apply(len)
    df['headline_word_count'] = df['headlines'].fillna('').astype(str).apply(lambda s: len(s.split()))
    df['description_length'] = df['description'].fillna('').astype(str).apply(len)
    df['description_word_count'] = df['description'].fillna('').astype(str).apply(lambda s: len(s.split()))
    df['content_length'] = df['content'].fillna('').astype(str).apply(len)
    df['content_word_count'] = df['content'].fillna('').astype(str).apply(lambda s: len(s.split()))
    return df


def plot_category_distribution(df):
    counts = df['category'].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', color='tab:blue', ax=ax)
    ax.set_title('Article Count by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    plt.tight_layout()
    out = OUTPUT_DIR / 'category_distribution.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved category distribution plot to', out)


def plot_length_distributions(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    df.boxplot(column=['headline_word_count'], by='category', ax=axes[0], grid=False)
    axes[0].set_title('Headline Word Count by Category')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Headline Word Count')
    df.boxplot(column=['description_word_count'], by='category', ax=axes[1], grid=False)
    axes[1].set_title('Description Word Count by Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Description Word Count')
    df.boxplot(column=['content_word_count'], by='category', ax=axes[2], grid=False)
    axes[2].set_title('Content Word Count by Category')
    axes[2].set_xlabel('Category')
    axes[2].set_ylabel('Content Word Count')
    fig.suptitle('Length Distribution Across Categories', fontsize=16)
    plt.tight_layout()
    out = OUTPUT_DIR / 'length_distribution_boxplots.png'
    fig.savefig(out)
    plt.close(fig)
    print('Saved length distribution plots to', out)


def normalize_text(text, remove_stopwords=True):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    else:
        tokens = [token for token in tokens if len(token) > 1]
    return ' '.join(tokens)


def tokenize_and_filter(text):
    return [token for token in normalize_text(text).split() if token not in STOP_WORDS and len(token) > 1]


def top_n_words(series, n=25):
    counts = Counter()
    for value in series:
        tokens = tokenize_and_filter(value)
        counts.update(tokens)
    return counts.most_common(n)


def save_top_words_report(df):
    with open(OUTPUT_DIR / 'top_words_by_category.md', 'w', encoding='utf-8') as out:
        out.write('# Top Words by Category\n\n')
        for category, subset in df.groupby('category'):
            out.write(f'## {category}\n\n')
            out.write('### Top words in content\n\n')
            for word, count in top_n_words(subset['content_clean']):
                out.write(f'- {word}: {count}\n')
            out.write('\n')
    print('Saved top words by category report to', OUTPUT_DIR / 'top_words_by_category.md')


def split_sentences(text):
    if pd.isna(text):
        return []
    text = str(text).replace('\n', ' ')
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [part.strip() for part in parts if part and part.strip()]


def normalize_sentence(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#[\w-]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def identify_redundant_sentences(df, text_column='content', output_path=None):
    sentence_entries = {}

    for row_idx, row in df.iterrows():
        text = row[text_column]
        for sentence in split_sentences(text):
            normalized = normalize_sentence(sentence)
            if len(normalized.split()) < REDUNDANT_SENTENCE_MIN_TOKENS:
                continue

            if normalized not in sentence_entries:
                sentence_entries[normalized] = []

            sentence_entries[normalized].append({
                'row_index': int(row_idx),
                'source_file': row.get('source_file', ''),
                'category': row.get('category', ''),
                'headline': row.get('headlines', ''),
                'original_sentence': sentence,
            })

    candidate_rows = []
    boilerplate_markers = [
        'follow us', 'follow on', 'read more', 'click here', 'subscribe', 'subscription',
        'newsletter', 'advertisement', 'advert', 'copyright', 'all rights reserved',
        'privacy policy', 'terms of use', 'facebook', 'twitter', 'instagram', 'google',
        'youtube', 'linkedin', 'tiktok', 'podcast', 'watch now', 'for more',
        'view the full', 'download the app', 'download our app', 'sign up',
    ]

    for normalized, entries in sentence_entries.items():
        freq = len(entries)
        if freq < REDUNDANT_SENTENCE_MIN_FREQUENCY:
            continue

        tokens = normalized.split()
        token_count = len(tokens)
        if token_count > REDUNDANT_SENTENCE_MAX_TOKENS:
            continue

        unique_ratio = len(set(tokens)) / max(1, token_count)
        lowered = normalized.lower()
        has_boilerplate_signal = any(marker in lowered for marker in boilerplate_markers)
        short_sentence = token_count <= 8
        low_information = unique_ratio <= REDUNDANT_SENTENCE_MIN_UNIQUE_RATIO
        score = 0
        if freq >= REDUNDANT_SENTENCE_MIN_FREQUENCY:
            score += 2
        if short_sentence:
            score += 1
        if low_information:
            score += 1
        if has_boilerplate_signal:
            score += 2
        if token_count <= 4:
            score += 1

        if score >= 3 or has_boilerplate_signal:
            sample_locations = []
            for entry in entries[:10]:
                sample_locations.append(
                    f"{entry['row_index']} ({entry['source_file']}, {entry['category']})"
                )
            candidate_rows.append({
                'normalized_sentence': normalized,
                'original_sentence': entries[0]['original_sentence'],
                'frequency': freq,
                'token_count': token_count,
                'unique_token_ratio': round(unique_ratio, 3),
                'has_boilerplate_signal': has_boilerplate_signal,
                'score': score,
                'example_locations': '; '.join(sample_locations),
                'example_headline': entries[0]['headline'],
            })

    report = pd.DataFrame(candidate_rows)
    if output_path is not None:
        report.sort_values(['frequency', 'score'], ascending=[False, False]).to_csv(output_path, index=False)
    return report


def remove_redundant_sentences(text, redundant_sentences):
    if pd.isna(text):
        return ''
    sentences = split_sentences(text)
    if not sentences:
        return ''
    cleaned_sentences = []
    for sentence in sentences:
        normalized = normalize_sentence(sentence)
        if normalized in redundant_sentences:
            continue
        cleaned_sentences.append(sentence)
    return ' '.join(cleaned_sentences).strip()


def preprocess_dataframe(df):
    df = df.copy()
    for column in ['headlines', 'description', 'content']:
        df[column] = df[column].fillna('').astype(str)
    df['category'] = df['category'].astype(str).str.strip().str.lower()

    df = compute_length_features(df)
    df['is_short_content'] = df['content_word_count'] < MIN_CONTENT_WORDS
    df['is_long_content'] = df['content_word_count'] > MAX_CONTENT_WORDS
    df['content_outlier'] = df['is_short_content'] | df['is_long_content']

    if DROP_OUTLIERS:
        before = len(df)
        df = df[~df['content_outlier']].copy()
        print(f'Dropped {before - len(df)} outlier rows based on content length.')

    redundant_report_path = OUTPUT_DIR / 'redundant_sentence_candidates.csv'
    redundant_report = identify_redundant_sentences(df, text_column='content', output_path=redundant_report_path)
    if not redundant_report.empty:
        print(f'Found {len(redundant_report)} candidate redundant sentences. Review report saved to {redundant_report_path}')
    else:
        print('No repeated low-information sentence candidates were found.')

    redundant_sentence_set = set(redundant_report['normalized_sentence'].tolist()) if not redundant_report.empty else set()
    if AUTO_REMOVE_REDUNDANT_SENTENCES and redundant_sentence_set:
        df['content_for_model'] = df['content'].apply(lambda x: remove_redundant_sentences(x, redundant_sentence_set))
        df['description_for_model'] = df['description'].apply(lambda x: remove_redundant_sentences(x, redundant_sentence_set))
        print('Removed detected redundant sentences from content and description fields before modeling.')
    else:
        df['content_for_model'] = df['content']
        df['description_for_model'] = df['description']

    df['headlines_clean'] = df['headlines'].apply(lambda x: normalize_text(x, remove_stopwords=True))
    df['description_clean'] = df['description_for_model'].apply(lambda x: normalize_text(x, remove_stopwords=True))
    df['content_clean'] = df['content_for_model'].apply(lambda x: normalize_text(x, remove_stopwords=True))
    df['text'] = (
        df['headlines_clean'] + ' ' +
        df['description_clean'] + ' ' +
        df['content_clean']
    ).str.strip()
    df['tokens'] = df['text'].apply(lambda x: x.split())
    return df


def build_vectorizer(df):
    if VECTORIZATION_METHOD == 'bow':
        vectorizer = CountVectorizer(
            strip_accents='unicode',
            stop_words='english',
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(df['text'])
        print('BoW matrix shape:', X.shape)
        print('Sample BoW feature names:', vectorizer.get_feature_names_out()[:20])
        return X, vectorizer

    if VECTORIZATION_METHOD == 'tfidf':
        vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            stop_words='english',
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(df['text'])
        print('TF-IDF matrix shape:', X.shape)
        print('Sample TF-IDF feature names:', vectorizer.get_feature_names_out()[:20])
        return X, vectorizer

    if VECTORIZATION_METHOD == 'embeddings':
        embedding_vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            stop_words='english',
            max_features=MAX_FEATURES,
            ngram_range=(1, 2),
        )
        X_sparse = embedding_vectorizer.fit_transform(df['text'])
        n_components = min(EMBEDDING_SIZE, X_sparse.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(X_sparse)
        print('Embedding matrix shape:', X.shape)
        np.save(OUTPUT_DIR / 'article_embeddings.npy', X)
        return X, {'vectorizer': embedding_vectorizer, 'svd': svd}

    raise ValueError(f'Unsupported vectorization method: {VECTORIZATION_METHOD}')


def main():
    df = load_data(CSV_FILES)
    data_report(df)
    df = preprocess_dataframe(df)

    print_section('Initial Exploratory Data Analysis')
    print('Headline length stats:')
    print(df['headline_word_count'].describe())
    print('\nDescription length stats:')
    print(df['description_word_count'].describe())
    print('\nContent length stats:')
    print(df['content_word_count'].describe())

    plot_category_distribution(df)
    plot_length_distributions(df)
    save_top_words_report(df)

    print_section('Data Pre-processing')
    print('Cleaned text sample:')
    print(df[['headlines_clean', 'description_clean', 'content_clean', 'text']].head(2).to_dict(orient='records'))

    print_section('Text Vectorization')
    print('Using vectorization method:', VECTORIZATION_METHOD)
    X, vectorizer = build_vectorizer(df)

    merged_out = OUTPUT_DIR / 'merged_preprocessed_data.csv'
    df.to_csv(merged_out, index=False)
    print('Saved merged preprocessed dataset to', merged_out)

    print_section('Milestone 1 Summary')
    print('Milestone 1 analysis completed. Outputs written to', OUTPUT_DIR)


if __name__ == '__main__':
    main()
