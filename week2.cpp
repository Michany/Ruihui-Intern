#include <iostream>
using namespace std;

int main(){
    float fahrenheit, celsius;
    cout << "Please enter the Fahrenheit temperature:"
    cin >> fahrenheit;

    celsius = (fahrenheit - 32) *5/9;

    cout.precision(3)
    cout << "The Celsius temperature is:" << celsius << endl;
    system("pause");
    return 0
}

