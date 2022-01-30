# Question Answering
Simply returning relevant documents is of limited use, we often want answers to our questions.We can factor this into two parts:
1. Information Retrieval
    - finding documents that (might) contain an answer
1. Reading Comprehension
    - finding an answer in a paragraph or a document

**Machine Comprehension (MC)**
![](img/a5c9bf5f.png)

## Stanford Question Answering Dataset (SQuAD)
- *extractive question answering*: answer must be a span in the passage.
- only span-based answers (no yes/no, counting, implicit why)

## Stanford Attentive Reader
Question representation and attention to start and end token:
![](img/1a47a105.png)
![](img/6816ff62.png)

> uses last n-representation

## Stanford Attentive Reader++
![](img/4a535add.png)
![](img/1b1030a6.png)

> uses representation concatenation

## Bi-Directional Attention Flow for Machine Comprehension (BiDAF)
![](img/fbbb5dfd.png)

Attention Flow layer: from the context to the question and from the question to the context:
![](img/0d2c322b.png)
![](img/792779ce.png)

## FusionNet
![](img/9b01af1f.png)
![](img/0c5997c2.png)

## ELMo and BERT preview
*Contextual word representations*
