# Parallelize k-Nearest Neighbors Search 🔍

Parallelize k-Nearest Neighbors Search is a program that constructs a [k-Dimensional tree](https://en.wikipedia.org/wiki/K-d_tree) and performs the [k-Nearest Neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) in parallel from training and query files. Designed a [thread-safe queue](https://www.educba.com/c-thread-safe-queue/) and [thread-pool](https://en.wikipedia.org/wiki/Thread_pool) to replicate a model similar to that of the producer-consumer design pattern to allow the program to concurrently building and querying the k-d tree. This repository also provides python script files that can generate training and query files for you to test the program on. This project was inspired by university course CS 447: High Performance Computing.

## Getting Started

These instructions will give you a copy of the neural network up and running on
your local machine for development and testing purposes.

### Prerequisites

To run this application locally on your computer, you'll need `Git`, `g++`, and `python` installed on your computer.

### Installing

Then run the following command in the command line and go to the desired directory to store this project:

Clone this repository:

    git clone https://github.com/JonathanCen/Parallized-k-Nearest-Neighbors-Search.git

Go into the cloned repository:

```
cd Parallized-k-Nearest-Neighbors-Search
```

Create training and query data:

```
make data
```

Compile cpp program:

```
make all
```

Run executable with the generated training and query file, and file to write the results:

```
./parallized-k-nn [n_cores] [training_file] [query_file] [result_file_name]
```

## More on generating query and training data

To generate a training file using `generate_training_file.py`:

```
python generate_training_file.py trainingCount cols dist
```

- trainingCount: the number of points to generate
- cols: the number of points to generate the number of dimensions for each point
- dist: the type of probability distribution to generate the points\*

Similarly, to generate a query file using `generate_query_file.py`:

```
python generate_training_file.py queryCount cols dist k
```

- trainingCount: the number of queries to generate
- cols: the number of dimensions for each point (should be the same as the training file)
- dist: the type of probability distribution to generate the points\*
- k: the number of neighbors to return for each query

\*dist param is the numerical value corresponding to the following distributions

0. Uniform Distribution
1. Centered Uniform Distribution
2. Beta Distribution
3. Exponential Distribution

## Contributing

All issues and feature requests are welcome.
Feel free to check the [issues page](https://github.com/JonathanCen/Parallized-k-Nearest-Neighbors-Search/issues) if you want to contribute.

## Authors

- **Jonathan Cen** - [LinkedIn](https://www.linkedin.com/in/jonathancen/), [Github](https://github.com/JonathanCen)

## License

Copyright © 2022 [Jonathan Cen](<ADD PERSONAL WEBSITE LINK>).\
This project is [MIT licensed](https://github.com/JonathanCen/Parallized-k-Nearest-Neighbors-Search/blob/main/LICENSE).
