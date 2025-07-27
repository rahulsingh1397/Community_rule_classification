# Jigsaw - Agile Community Rules Classification

A machine learning solution for predicting Reddit comment rule violations with a focus on generalization to unseen rules.

## 🎯 Competition Overview

- **Task**: Binary classification to predict whether Reddit comments violate specific subreddit rules
- **Challenge**: Training data contains only 2 rules, but test data includes unseen rules
- **Evaluation**: Column-averaged AUC
- **Deadline**: October 23, 2025

## 📊 Dataset Characteristics

- **Training Data**: 2,029 samples across 2 rules (~1,000 each)
- **Rules**: 
  1. No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed
  2. No Legal Advice: Do not offer or request legal advice
- **Class Balance**: 50.8% violation rate (well balanced)
- **Key Challenge**: Generalize from 2 training rules to unknown number of test rules

## 🚀 Strategy & Approach

### Core Challenge
The primary challenge is **rule generalization** - learning transferable violation patterns that work across different rule types rather than memorizing rule-specific patterns.

### Key Insights from Analysis
1. **Violations are longer**: Average 37+ more characters than non-violations
2. **Domain-specific vocabulary matters**: Legal, commercial, promotional terms
3. **Text structure patterns**: Sentence count, formatting indicators
4. **Transferable features**: Focus on semantic content over rule-specific keywords

## 📁 Project Structure

```
├── 01_EDA.ipynb                    # Exploratory Data Analysis
├── 03_rule_analysis.ipynb          # Deep rule pattern analysis
├── 04_baseline_transferable.ipynb  # Enhanced baseline with transferable features
├── simple_data_check.py            # Quick data analysis script
├── requirements.txt                # Python dependencies
├── EDA_insights.txt                # Key findings summary
└── README.md                       # This file
```

## 🔧 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rahulsingh1397/Community_rule_classification.git
   cd Community_rule_classification
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv jigsaw_env
   jigsaw_env\Scripts\activate  # Windows
   # or
   source jigsaw_env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download competition data**:
   - Place `train.csv` and `test.csv` in `Data/` directory
   - Ensure Kaggle API is configured for data access

## 🧪 Methodology

### 1. Exploratory Data Analysis (01_EDA.ipynb)
- Dataset structure and distribution analysis
- Class balance and rule-specific patterns
- Text characteristics and violation rates

### 2. Rule Pattern Analysis (03_rule_analysis.ipynb)
- Deep dive into the 2 training rules
- Cross-rule pattern discovery
- Transferable feature engineering
- Semantic analysis with embeddings

### 3. Transferable Feature Engineering
Key features that generalize across rule types:
- **Text statistics**: Length, word count, sentence structure
- **Commercial indicators**: Buy, sell, price, promotional language
- **Advice patterns**: Should, must, recommend, suggest
- **Legal terms**: Lawyer, court, law, legal advice
- **Formatting patterns**: Punctuation, URLs, special characters

### 4. Enhanced Baseline Model
- Rule-agnostic preprocessing
- Transferable feature set based on analysis
- Cross-validation strategy for generalization testing
- TF-IDF + LightGBM baseline with transferable features

## 📈 Key Findings

### Most Discriminative Cross-Rule Features:
1. **char_count/word_count**: Violations are consistently longer
2. **legal_words**: Strong predictor (0.346 correlation)
3. **commercial_words**: Higher in violations
4. **promotional_words**: Promotional language patterns
5. **sentence_count**: Text structure matters

### Transferable Violation Indicators:
- **Length patterns**: Violations tend to be more detailed/longer
- **Domain vocabulary**: Professional, commercial, legal terms
- **Urgency language**: Immediate, urgent, quickly, ASAP
- **Instructional tone**: How-to, guide, tutorial patterns

## 🎯 Competition Strategy

1. **Focus on Generalization**: Build models that capture violation concepts rather than rule-specific patterns
2. **Transferable Features**: Emphasize semantic content over exact keyword matching
3. **Meta-Learning**: Design approaches that can quickly adapt to new rule types
4. **Ensemble Methods**: Combine multiple approaches for robust performance

## 🔬 Future Work

- [ ] BERT/RoBERTa fine-tuning with rule-aware attention
- [ ] Meta-learning approaches (MAML, Prototypical Networks)
- [ ] Advanced ensemble methods
- [ ] Rule embedding techniques using provided examples
- [ ] Data augmentation for better generalization

## 📊 Performance Targets

- **Baseline (TF-IDF + LightGBM)**: 0.82-0.85 AUC
- **Enhanced Transferable Model**: 0.87-0.90 AUC
- **Competition Goal**: Top 10% (estimated 0.92+ AUC)

## 🤝 Contributing

This project is part of the Jigsaw Agile Community Rules Classification Kaggle competition. Feel free to explore the notebooks and analysis for insights into rule generalization strategies.

## 📝 License

This project is for educational and competition purposes.

---

**Note**: This project focuses on the challenging problem of generalizing from limited training rules to unseen test rules, making it an excellent case study in transfer learning and rule generalization for NLP tasks.
