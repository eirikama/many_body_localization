
# Compiler and flags
CXX = g++
CXXFLAGS = -I/usr/include/eigen3
LIBS = -llapacke -llapack -lopenblas

# Target executable
TARGET = solve_random_heisenberg

# Source files
SRC = main.cpp

# Default target
all: $(TARGET)

# Compile target
$(TARGET): $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) $(LIBS) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET)
