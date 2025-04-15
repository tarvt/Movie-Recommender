# Content-Based Movie Recommendation System

In this project, the recommendation process leverages a vector-based approach to model movies, where each movie is represented as a vector of 100 items that encapsulate various features and characteristics(The process of creating these vectors is explained in the `vectorize_movies.ipynb` notebook.). This vectorization enables the system to assess similarity among movies by calculating the distance between their respective vectors in a multidimensional space.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries specified in `requirements.txt`

### Download Dataset

To get started, download the movies dataset from Kaggle:
[Download Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
After downloading the dataset, place it in the `dataset` folder of the project.

## How to Use

### Running the Recommendation System

To run the recommendation system, use the following command in your terminal:

```bash
python .\recommendation_system.py
```

## Input and Output Examples

You will be prompted to enter a movie ID to get similar movie IDs. Here are some example movie IDs you can use:

- **ID 862** : Toy Story
- **ID 4584** : Sense and Sensibility
- **ID 9603** : Clueless
- **ID 807** : Se7en

The output guides the user to enter a specific movie ID.
It recommends similar movies based on the input, each with a unique ID and a brief description to help the user understand what the movie is about.

### Example Interaction

## Future Usages

In future updates, we can extend our model to incorporate user preferences. Each user can be modeled as the sum of their favorite movies, weighted by their scores for those movies. This will enhance the recommendation system's accuracy and personalization.
