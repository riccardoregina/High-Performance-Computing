CC = nvcc
CFLAGS = -O3
TARGET = trapezio.out
OBJS = main.o trapezio.o utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cu trapezio.h utils.h
	$(CC) $(CFLAGS) -c main.cu -o main.o

trapezio.o: trapezio.cu trapezio.h
	$(CC) $(CFLAGS) -c trapezio.cu -o trapezio.o

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) -c utils.cpp -o utils.o

clean:
	rm -f *.o $(TARGET)