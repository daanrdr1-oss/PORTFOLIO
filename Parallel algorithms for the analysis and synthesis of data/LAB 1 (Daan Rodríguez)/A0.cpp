#include <iostream>
#include <string>
#include <sstream>

int contarpalabras(const std::string &input) {
	
    std::stringstream iss(input);
    std::string palabra;
    int contador = 0;
    while (iss >> palabra) {
        contador++;
    }
    return contador;
    
}

int main(int argc, char *argv[]) {

    std::string input = argv[1];
    int contador = contarpalabras(input);
    std::cout << "Number of words: " << contador << std::endl;
    return 0;
    
}


   