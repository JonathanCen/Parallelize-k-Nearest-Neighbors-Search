TARGET = parallized-k-nn

CC = g++
CFLAGS = -g -Wall -Wextra -pedantic -pthread

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) $(TARGET).cpp -o $(TARGET)

data:
	python generate_training_file.py 1000000 3 0
	python generate_query_file.py 50 3 0 4

clean:
	rm -rf  $(TARGET)

clean_data:
	rm -rf	*.dat