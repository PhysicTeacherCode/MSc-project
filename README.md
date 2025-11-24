# Replicating *The Statistical Mechanics of Twitter Communities* Using the Bluesky Social Network

This repository is an attempt to reproduce — and adapt — the results of the article:

> **Gavin Hall and William Bialek (2019)** — *The Statistical Mechanics of Twitter Communities*
> DOI: **10.1088/1742-5468/ab3af0**

The main goal is to investigate whether the same statistical properties observed in Twitter communities also emerge in the **Bluesky** social network.

This is the [link](https://drive.google.com/drive/folders/1w2yRyMeoPIUw0AFl0sN3HXia6j4HNqm-?usp=sharing) for data collected so far.

---

## 📌 Motivation

The original paper uses statistical mechanics tools to analyze user interactions, uncovering patterns such as:

* Community formation
* Correlations between users
* Network structure
* Distribution of influence and connectivity

With the rise of Bluesky and its open API, this project explores whether similar phenomena arise on a more decentralized platform — or if new behavioral dynamics emerge.

---

## 🎯 Project Objectives

* Collect public Bluesky interaction data.
* Build network graphs (followers, posts and interactions).
* Apply methods inspired by the paper, such as:

  * Pairwise correlation matrices
  * The Maximun Entropy models for social interactions
* Investigate how structural differences between Twitter and Bluesky communities affect the statistical outcomes.

---

## 🛠️ Tools & Technologies

* **Python 3.x**
* **Bluesky API**
* Main libraries:

  * `pandas`
  * `matplotlib` / `seaborn`
  * `requests`
  * `time`
  * `concurrent.futures` (for time optmization)
* Jupyter Notebooks for analysis workflows.

---

## 📊 Data

Data is collected directly from the public Bluesky API.
Depending on Bluesky’s data policies, *raw data may not be included* and may instead be reconstructed via provided scripts.

---

## 📁 Repository Structure

```
In progress
```

---

## 📚 Main Reference

Gavin Hall and William Bialek. (2019).
**The statistical mechanics of Twitter communities**.
*Journal of Statistical Mechanics: Theory and Experiment*.
DOI: 10.1088/1742-5468/ab3af0

---

## 🚧 Project Status

> **In progress.**
> Data collection, graph construction, and initial analyses are ongoing.

---

## 🤝 Contributing

Contributions are welcome!
Suggestions, improvements, alternative methods, or discussions about the original paper are especially encouraged.

---

## 📄 License

```
In progress
```

---

## ✨ Final Note

This project aims to bridge statistical physics and data science applied to modern decentralized social networks.
If you're interested in complex networks, statistical mechanics, or social platforms like Bluesky, this repository may be useful to you.
