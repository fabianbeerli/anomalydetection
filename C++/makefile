
CC  = g++ -Wall -O3 -fopenmp
FILE = $(CURDIR)
SRC = $(FILE)/src
OBJ = $(FILE)/obj
IND = $(FILE)/include/
INC = -I $(IND)

SOURCEFILES = $(wildcard $(SRC)/*.cpp)
OBJECTFILES = $(patsubst $(SRC)%.cpp,$(OBJ)%.o,$(SOURCEFILES))


all: $(OBJECTFILES)

$(OBJ)%.o: $(SRC)%.cpp | $(OBJ)
	$(CC) $(INC) -c $< -o $@
    
$(OBJ):
	mkdir -p $@

clean:
	rm $(OBJECTFILES)	

# Compile the executable
aida_example: $(OBJ) tests/aida_example.cpp
	$(CC) $(INC) $(OBJ) tests/aida_example.cpp -o bin/aida_example
