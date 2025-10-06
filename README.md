# End-to-End Entity Resolution with Snowflake Cortex AI

![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

A production-ready entity resolution solution leveraging Snowflake's native AI capabilities to automatically match and reconcile product data across different retail catalogs.

---

## 📋 Table of Contents

- [Overview](#-Overview)
- [The Business Problem](#-the-business-problem)
- [Solution Architecture](#solution-architecture) 
- [Prerequisites](#-prerequisites)
- [Data Files](#data-files)
- [Key Features](#-key-features)
- [Documentation](#-documentation)
- [Authors](#-authors)
- [Quick Links](#-quick-links)

---

## 🎯 Overview

This repository contains a complete end-to-end entity resolution pipeline that demonstrates how to match product records across different datasets with varying schemas, naming conventions, and data quality. Built entirely within Snowflake using Cortex AI, Streamlit, and Notebooks, this solution achieves **85%+ accuracy** while requiring minimal manual intervention.

**What makes this solution unique:**
- 🤖 **AI-Powered**: Uses Snowflake Cortex AI for schema mapping and intelligent matching
- ⚡ **Hybrid Approach**: Combines vector similarity (fast) with AI_CLASSIFY (smart) for optimal performance
- 🎨 **Interactive UIs**: Built with Streamlit in Snowflake for seamless user experience
- 📊 **Production-Ready**: Includes audit trails, confidence scoring, and quality metrics
- 💰 **Cost-Efficient**: AI_CLASSIFY used only when needed (~30% of cases)

---

## 🔍 The Business Problem

Imagine you're a merchandising manager at **ABT Electronics** and need to perform competitive pricing analysis against **Best Buy**. You've scraped Best Buy's product data, but face a critical challenge:

**How do you match Best Buy's products to your own catalog when:**
- Column names are completely different (`SKU` vs `PRODUCTID`, `PRODUCT_LABEL` vs `NAME`)
- Product descriptions vary significantly
- No common identifiers exist between datasets
- Manual mapping of 1,000+ products is impractical

This is a classic **entity resolution** problem - determining which records in different datasets refer to the same real-world entity.

### Traditional Challenges
- ❌ Manual mapping takes weeks and is error-prone
- ❌ Rule-based systems are brittle and don't scale
- ❌ Custom ML models require training data and maintenance
- ❌ Multiple tools and technologies increase complexity

---

## Solution Architecture

This solution implements a **three-stage workflow** that progressively refines matches from raw data to validated results:

### Stage 1: Data Harmonization
**Purpose**: Transform disparate schemas into unified datasets

- 📊 **Schema Profiling**: Analyze column types, cardinality, and sample values
- 🧠 **AI-Powered Mapping**: Use Cortex AI (mistral-large) to recommend semantic field mappings
- ✏️ **Interactive Review**: User-friendly interface to validate and adjust AI recommendations
- 🔄 **Dataset Creation**: Generate harmonized tables with consistent structure

**Output**: Standardized datasets ready for matching

### Stage 2: Hybrid Entity Matching
**Purpose**: Intelligently match products using a two-stage approach

- 🎯 **Vector Embeddings**: Generate semantic embeddings using `EMBED_TEXT_1024` (voyage-multilingual-2)
- ⚡ **Fast Path**: High-confidence matches (≥80% similarity) resolved via vector comparison
- 🤖 **Smart Path**: Ambiguous cases use `AI_CLASSIFY` to select best match from top candidates
- 📈 **Confidence Scoring**: Enhanced scoring based on match method and similarity

**Output**: Matched and unmatched record tables with confidence scores

### Stage 3: Unmatched Record Reconciliation
**Purpose**: Human-in-the-loop validation for uncertain matches

- 🔍 **Candidate Review**: View top 20 candidates per unmatched record with similarity scores
- ✅ **Selective Approval**: Choose correct matches or confirm "no match found"
- 📄 **Pagination & Search**: Efficiently navigate through hundreds of records
- 💾 **Batch Processing**: Move approved matches to final matched table

**Output**: Validated, production-ready entity matches

---


### File Descriptions

| File | Type | Purpose |
|------|------|---------|
| `Data Harmonization with Snowflake Cortex AI.py` | Streamlit App | Schema analysis and field mapping with AI recommendations |
| `HYBRID_ENTITY_MATCHING_WITH_AI_CLASSIFY.ipynb` | Snowflake Notebook | Vector similarity + AI_CLASSIFY matching algorithm |
| `Entity Resolution - Unmatched Records.py` | Streamlit App | Interactive review interface for uncertain matches |
| `Abt.csv` | Data | ABT Electronics product catalog |
| `Buy.csv` | Data | Best Buy product catalog |
| `abt_buy_perfectMapping.csv` | Data | Ground truth mapping for accuracy evaluation |

---

## ✅ Prerequisites

### What You'll Need

- A [Snowflake](https://signup.snowflake.com/) account. Sign up for a 30-day free trial account, if required.
- Access to [Snowflake Cortex AI functions](https://docs.snowflake.com/user-guide/snowflake-cortex/aisql) (available in most commercial regions)
- Basic understanding of [Snowflake Notebooks](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks) and [Streamlit in Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)

**Knowledge Prerequisites:**
- Basic SQL proficiency
- Familiarity with Python and Pandas
- Understanding of data integration concepts

---

## 🚀 To get started



1. **Download the neccesary files**
   
   The 3 source files (2 streamlit apps and 1 notebook) can be found [here](https://github.com/sfc-gh-jrauh/sfentityresolution).
  

2. **Use this Snowflake Quickstart Guide as a rereference**

3. **Follow along with this video**
  

## Data Files

>[https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution]
>
>**Scroll down to find the Abt-Buy dataset link.**

> **Note on the Datasets Used:** The datasets used in this quickstart are commonly used entity resolution test datasets. These datasets are made available by the database group of Prof. Erhard Rahm under the [Creative Commons license](https://creativecommons.org/licenses/by/4.0/). Column titles are changed at the table level from the original CSV files.
>
> **Citation:** Hanna Köpcke, Andreas Thor, and Erhard Rahm. 2010. Evaluation of entity resolution approaches on real-world match problems. Proc. VLDB Endow. 3, 1–2 (September 2010), 484–493. [https://doi.org/10.14778/1920841.1920904](https://doi.org/10.14778/1920841.1920904)
---

## ✨ Key Features

### 🎯 High Accuracy
- **85-90%** match accuracy against ground truth
- Confidence-based thresholding for quality control
- Validation metrics at every stage

### ⚡ Performance Optimized
- Vector similarity handles 70-80% of matches instantly
- AI_CLASSIFY used strategically for ambiguous cases
- Processes 1,000+ products in under 5 minutes

### 💰 Cost Efficient
- AI_CLASSIFY invoked only when multiple viable candidates exist
- Typical usage: 20-40% of records
- Minimal compute resources for vector operations

---

## 📚 Documentation

### Official Snowflake Documentation
- [Snowflake Cortex AI Overview](https://docs.snowflake.com/en/user-guide/snowflake-cortex/overview)
- [Vector Embeddings with EMBED_TEXT_1024](https://docs.snowflake.com/en/user-guide/snowflake-cortex/vector-embeddings)
- [AI_CLASSIFY Function](https://docs.snowflake.com/en/user-guide/snowflake-cortex/ml-functions/classification)
- [Streamlit in Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- [Snowflake Notebooks](https://docs.snowflake.com/en/user-guide/ui-snowsight-notebooks)

### Related Resources
- [Vector Similarity Search in Snowflake](https://docs.snowflake.com/en/user-guide/querying-vector-data)
- [Snowflake Cortex Best Practices](https://docs.snowflake.com/en/user-guide/snowflake-cortex/overview#best-practices)

---



## 👥 Authors

**Joshua Rauh** and **Ben Marzec**  


- 📧 Email: [Joshua.Rauh@Snowflake.com] / [Ben.Marzec@Snowflake.com]


---



## 🔗 Quick Links

- 📖 [Snowflake Quickstart Guide](#) *(Link TBD)*
- 💻 [Snowflake Free Trial](https://signup.snowflake.com/)
- 🎓 [Snowflake University](https://learn.snowflake.com/)
- 💬 [Snowflake Community](https://community.snowflake.com/)
- 📺 [Snowflake YouTube Channel](https://www.youtube.com/user/snowflakecomputing)

---
