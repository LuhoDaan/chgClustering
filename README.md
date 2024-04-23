# Automatic recognition of recurrent issues

Problem of recurrent emergencies. Need of an automatic tool for clustering common emergencies, without the need of having the releant knowledge of recognizing the common emergencies and without having to read them one by one.

## Solution
Implemented an information retrieval problem as if it was a clusteriung problem. Meaning starting from the vectorization and cleaning of the raw text, iteratively it has been looked for all the similar emergencies to each document by means of a matrix of similarity scores that was obtained thorugh the cosine similarity formula.
The vectorization was done by means of tf-idf.
