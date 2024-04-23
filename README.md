# RAG-InLAvS IR Project: A Legal Chatbot for India

## Introduction
Welcome to the RAG-InLAvS (RAG Based Indian Legal Advisory System) project! This initiative aims to develop a chatbot tailored to the Indian legal landscape, providing accessible and intelligent legal advice to users. This README file serves as a guide to understand the motivation, technical approach, data processing, evaluation metrics, and future work associated with the project.

## Motivation
The RAG-InLAvS project is driven by the following motivations:
- **Bridging the Legal Access Gap:** India faces challenges in accessing legal aid. This chatbot offers a scalable solution to provide legal information readily.
- **Focus on Indian Audience:** Tailored specifically to Indian laws, it addresses issues commonly faced by Indian users.
- **Affordability and Convenience:** Compared to traditional methods, the chatbot offers a cost-effective and convenient way to obtain legal information.

## Novelty
Key features that distinguish RAG-InLAvS from other legal chatbots include:
- **Indian Context:** Focusing on legal issues relevant to Indian users.
- **Machine Feedback Mechanism:** Utilizing a custom QnA dataset for learning and improving responses.
- **Diverse Datasets:** Incorporating two datasets, LegalActsDB and r/PileofLaw, for context generation and response fine-tuning.

## Technical Approach
The technical approach employed in RAG-InLAvS includes:
- **RAG (Retrieval Augmented Generation):** Combining information retrieval and language generation techniques.
- **Efficient Embedding and Retrieval:** Utilizing BM25 with N-grams for accurate context understanding and relevance scoring.
- **User Feedback Integration:** Learning from user feedback to enhance responses.

## Data Processing and Analysis
Methods employed for data processing and analysis:
- **Chunking:** Breaking down datasets into smaller chunks for efficiency and scalability.
- **Error Removal:** Applying text normalization and spelling correction techniques for data quality assurance.
- **IR Retrieval:** Using techniques like BM25 with N-grams for efficient information retrieval.

## Classification and Feedback
Utilization of classification and feedback mechanisms:
- **Data Generation:** Training a classifier to distinguish real from artificially generated data.
- **RLHF (Reinforcement Learning from Human Feedback):** Employing a pipeline to enhance model alignment and accuracy using human feedback.

## Evaluation
Evaluation of the model's performance includes:
- **Metrics:** Utilizing Hallucination Score, Context Adherence Score, and Generation Score against benchmarks like LegalGPT and ChatGPT 3.5.
- **Results:** Demonstrating promising performance, approaching benchmark levels with significantly less data.

## Future Work
Plans for future development include:
- **Short Term:** Improving handling of legal document complexities, enhancing embedding techniques, and optimizing the tokenizer.
- **Medium Term:** Generating diverse human-like text responses and training a feedback module for further alignment.
- **Long Term:** Continuously incorporating user feedback to improve user-friendliness and align model results with user expectations.

## Conclusion
RAG-InLAvS offers a promising approach to provide accessible and accurate legal information to the Indian population. With its combination of diverse datasets, efficient retrieval techniques, and machine learning, the project showcases the potential for effective legal chatbots tailored to specific contexts. Further development and user feedback integration are expected to enhance the model's value for legal empowerment in India.

## Commands
```bash
git clone "https://github.com/DevanshArora-2002/IR-Project.git"
```

## Appendix
- **Research Paper URL:** https://www.overleaf.com/project/65c1f64b1fc254137181e6d3
- **Powerpoint URL:** https://www.canva.com/design/DAF_rQjo5rM/saDTJLIjPRci3RAi1lb7cQ/edit?utm_content=DAF_rQjo5rM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
- **Github Repository:** https://github.com/DevanshArora-2002/IR-Project.git
- **Youtube URL:** https://www.youtube.com/watch?v=QX_s909PF1g
- **Website URL:** The website is hosted locally.

Thank you for your interest in the RAG-InLAvS IR Project! For more information or inquiries, please contact Group-17.