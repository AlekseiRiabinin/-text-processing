# NLP: Toxic Comment Classification

## 1. Description
Platforms that aggregate user content are the foundation of knowledge sharing on the Internet such as blogs, forums and discussion boards. But the catch is that not all people on the Internet are interested in participating nicely, and some see it as an avenue to vent their rage, insecurity, and prejudices. Web apps run on user generated content, and are dependent on user discussion to curate and approve content. The problem with this is that people will frequently write things they should not, and to maintain a positive community this toxic content and the users posting it need to be removed quickly. 

## 2. Application
Digital Agencies / Marketing / PR / SEO, Ads Platforms, Internet Services, Online Stores, Business Services [b2b] (outsourcing consulting audit), Startups, IT Company

## 3. Tech stack
- PyTorch
- PySpark
- Sklearn
- NLTK
- Pandas
- Numpy
- Matplotlib
- Seaborn

## 4. Results
Due to the PyTorch model having pretrained weights, the filter on toxic words works well. Nevertheless, it has to be run on GPU, so the f1-score on CPU is average (0.48). Better score demonstrates a model built in Sklearn using TF-IDF method (f1=0.78). Additionally, this approach provides various ways to adjust the model parameters and get to the appropriate outcome on CPU relatively fast.
