
# Text2Image Search System

## Objective
The Text2Image Search project is designed to leverage machine learning and vector search technologies to enable searching for similar images based on textual queries. This system aims to showcase the ability to understand and implement fundamental concepts in machine learning and software development, focusing on practical applications in image retrieval.

## Dataset
This project utilizes the Advertisement Image Dataset for demonstrating the text-to-image search capability. The dataset provides a diverse collection of advertisement images, suitable for evaluating the performance of the search system across various textual queries. This choice ensures a practical application scenario, simulating real-world usage in e-commerce and digital marketing contexts.

## Implementation
The system is built with the following components:
- **Data Loader (`data_loader.py`)**: Prepares and processes the image dataset for embedding extraction.
- **Model (`model.py`)**: Utilizes the CLIP model from OpenAI for converting images and text queries into embeddings.
- **Indexing (`index_embeddings_in_qdrant.py`)**: Indexes the image embeddings in Qdrant, a vector search engine, enabling efficient similarity search.
- **Streamlit App (`streamlit_app.py`)**: A user-friendly interface for performing text-to-image searches and displaying results.

## How to Run
1. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare the dataset and extract embeddings using `data_loader.py`.
   (Note:For running this project, place the Advertisement Image Dataset in the `data/subfolder-0/0` directory within your project folder. Ensure this dataset is structured properly for the data loader to process it effectively. After placing the dataset, run the provided Python scripts in sequence to generate image embeddings.)
5. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Query Evaluation
1. Our Text-to-Image Search system was evaluated using a variety of descriptive queries to showcase its capabilities. Successful queries like "a blue sky with fluffy white clouds","woman running","vintage car advertisement","Fashion clothing sale", "home appliance sale" and many other prompts returned highly relevant images as shown below, demonstrating the system's ability to accurately interpret and match textual queries with appropriate images from the Advertisement Image Dataset.

2. However, more abstract queries such as "Sustainable living practices" or "Cryptocurrency investment opportunities" occasionally returned less relevant results, but as far as I have tested it gave excellent results in 90 percent of the queries and in other cases indicating a potential area for improvement in understanding nuanced or less visually distinct concepts or if the data may not be present in the dataset.

3. To quantitatively evaluate retrieval accuracy, we suggest a method involving the labeling of a subset of the dataset with specific tags corresponding to the content of the images. These tags can then be used as queries to retrieve images, and the accuracy can be measured based on the relevance of the returned images to the tags. This approach would provide a systematic way to assess and refine the search system's performance.

Below queries which worked really well:
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/d9b61156-5748-4d86-8f03-c5a550949479)
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/aba4275e-2c22-4bfd-9fa4-cb64e88ddab5)
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/87049cad-5453-4285-b0ea-ac11ebb5d20f)
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/6f4efa7b-2dbf-47bc-9803-353c271a253c)
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/00629c87-3c25-4fe5-9d7b-3bf8420ee1d7)

Below queries which were not perfectly accurate: 
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/3c6dde3b-b09b-486b-b3d8-5d5c6a4a416f)
![image](https://github.com/V-Krishna-Chaitanya/Text2Image_Search/assets/102852229/813f82c0-14fd-4ff1-937a-5e63de99b6b8)




## Challenges and Improvements
- I faced challenges while pushing the embeddings to the quadrant database and then reached out to the support team to get it corrected. Thanks for the support team. There was some challenging about deciding the model for converting the images to embeddings and finally decided to leverage the CLIP model which is a start-of-the-model in learning the sematics of the images
- Another challenge could be the scalability of the system as the dataset grows. Potential avenues for improvement include enhancing the model's ability to understand and process abstract queries through advanced natural language processing techniques, increasing the diversity of the dataset to cover a wider range of concepts and scenarios, and implementing more sophisticated vector search algorithms to improve search accuracy and speed. 
